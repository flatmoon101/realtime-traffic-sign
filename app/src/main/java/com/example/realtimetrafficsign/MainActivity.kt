package com.example.realtimetrafficsign

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.realtimetrafficsign.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var camera: Button
    private lateinit var gallery: Button
    private lateinit var realtime: Button
    private lateinit var imageView: ImageView
    private lateinit var result: TextView
    private val imgSize = 150

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)

        camera = findViewById(R.id.button_camera)
        gallery = findViewById(R.id.button_gallery)
        realtime = findViewById(R.id.button_realtime)
        result = findViewById(R.id.hasil)
        imageView = findViewById(R.id.img)

        camera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 3)
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

        gallery.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, 1)
        }

        realtime.setOnClickListener {
            val intent = Intent(this, InferenceActivity::class.java)
            startActivity(intent)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun classifyImage(image: Bitmap) {
        try {
            val model = Model.newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imgSize * imgSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0
            for (i in 0 until imgSize) {
                for (j in 0 until imgSize) {
                    val value = intValues[pixel++] // RGB
                    byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255))
                    byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255))
                    byteBuffer.putFloat((value and 0xFF) * (1f / 255))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidences = outputFeature0.floatArray
            // Find the index of the class with the biggest confidence
            var maxPos = 0
            var maxConfidence = 0f
            for (i in confidences.indices) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i]
                    maxPos = i
                }
            }
            val classes = arrayOf("Putar Balik", "Dilarang Putar Balik", "Belok Kiri", "Belok Kanan")

            if(maxConfidence > 0.96) {
                result.text = "Hasil: ${classes[maxPos]} (%.2f%%)".format(maxConfidence * 100)
            } else {
                result.text = "Hasil: Tidak diketahui. Coba gambar yang lebih jelas!"
            }
            // Releases model resources if no longer used
            model.close()
        } catch (e: IOException) {
            Log.e("Model Error", "Error loading the model", e)
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                3 -> {
                    val image = data?.extras?.get("data") as Bitmap
                    val dimension = image.width.coerceAtMost(image.height)
                    var croppedImage = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                    imageView.setImageBitmap(croppedImage)

                    croppedImage = Bitmap.createScaledBitmap(croppedImage, imgSize, imgSize, false)
                    classifyImage(croppedImage)
                }

                1 -> {
                    val uri: Uri? = data?.data
                    try {
                        var image = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                        imageView.setImageBitmap(image)

                        image = Bitmap.createScaledBitmap(image, imgSize, imgSize, false)
                        classifyImage(image)
                    } catch (e: IOException) {
                        e.printStackTrace()
                    }
                }
            }
        }
    }
}