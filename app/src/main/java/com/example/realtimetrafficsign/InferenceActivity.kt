package com.example.realtimetrafficsign

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.realtimetrafficsign.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class InferenceActivity : AppCompatActivity() {

    private lateinit var resultTextView: TextView
    private lateinit var cameraExecutor: ExecutorService
    private val imgSize = 150

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.inference_activity)

        resultTextView = findViewById(R.id.resultTextView)

        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }

        // Start CameraX
        startCamera()

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetRotation(Surface.ROTATION_0)
                .build()
                .also {
                    it.surfaceProvider = findViewById<PreviewView>(R.id.previewView).surfaceProvider
                }

            val imageCapture = ImageCapture.Builder()
                .setTargetRotation(Surface.ROTATION_0)
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(Surface.ROTATION_0)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        analyzeImage(image, true)
                        image.close()
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e("CameraX", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeImage(image: ImageProxy, isRotated: Boolean = false) {
        var bitmap = imageProxyToBitmap(image)

        if(isRotated) {
            bitmap = rotateBitmap(bitmap, 90)
        }
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap , imgSize, imgSize, false)
        val normalizedByteBuffer = normalizeByteBuffer(scaledBitmap)

        classifyImage(normalizedByteBuffer)
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }


    private fun normalizeByteBuffer(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)

        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatBuffer = ByteBuffer.allocateDirect(4 * width * height * 3)
        floatBuffer.order(ByteOrder.nativeOrder())

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            floatBuffer.putFloat(r)
            floatBuffer.putFloat(g)
            floatBuffer.putFloat(b)
        }

//        return Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565)
        return floatBuffer
    }

    private fun classifyImage(image: ByteBuffer) {
        val model = Model.newInstance(applicationContext)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imgSize, imgSize, 3), DataType.FLOAT32)
//        val byteBuffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3).apply {
//            order(ByteOrder.nativeOrder())
//        }
//
        inputFeature0.loadBuffer(image)

        // Runs model inference and gets result
        val outputs = model.process(inputFeature0)
        val confidences = outputs.outputFeature0AsTensorBuffer.floatArray
        val classes = arrayOf("Putar Balik", "Dilarang Putar Balik", "Belok Kiri", "Belok Kanan")

        val maxIndex = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val confidence = confidences[maxIndex]

        runOnUiThread {
            resultTextView.text = if (confidence > 0.96) {
                "Hasil: ${classes[maxIndex]} (%.2f%%)".format(confidence * 100)
            } else {
                "Hasil: Tidak jelas. Coba lagi."
            }
        }

        model.close()
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Y
        yBuffer.get(nv21, 0, ySize)

        // U and V are swapped
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, out)
        val byteArray = out.toByteArray()

        val bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)!!

        // Konversi ke format RGB jika Bitmap berformat ARGB
        return bitmap.copy(Bitmap.Config.RGB_565, true) // Menghapus alpha channel
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
