package com.arm.armnn_demo;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Size;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("native-lib");
    }

    private static final int PERMISSIONS_REQUEST_CODE = 1;
    private static final String CAMERA_PERMISSION = "android.permission.CAMERA";

    private PreviewView previewView;
    private TextView outputTextView;
    private Classifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = (PreviewView)findViewById(R.id.previewView);
        outputTextView = (TextView)findViewById(R.id.outputTextView);
        classifier = new Classifier(getAssets());

        if(arePermissionsGranted()) {
            previewView.post(() -> startCamera());
        }
        else {
            requestPermissions();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> providerFuture = ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            try {
                bindCamera(providerFuture.get());
            } catch (ExecutionException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCamera(ProcessCameraProvider cameraProvider) {
        DisplayMetrics metrics = new DisplayMetrics();
        previewView.getDisplay().getRealMetrics(metrics);

        Size previewSize = new Size(metrics.widthPixels, metrics.widthPixels);
        Size analysisSize = new Size(Classifier.INPUT_WIDTH, Classifier.INPUT_HEIGHT);

        Preview preview = new Preview.Builder()
                .setTargetResolution(previewSize)
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setTargetResolution(analysisSize)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        analysis.setAnalyzer(Executors.newSingleThreadExecutor(), image -> classify(image));

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis);
        preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));
    }

    private void classify(ImageProxy image) {
        long startTime = System.currentTimeMillis();

        Bitmap bitmap = proxyToBitmap(image);
        image.close();
        String prediction = classifier.predict(bitmap);

        long frameTime = System.currentTimeMillis() - startTime;
        float fps = 1000.0f / frameTime;

        long inferenceTime = classifier.getLatestInferenceTime();

        runOnUiThread(() -> {
            outputTextView.setText(createOutputText(prediction, fps, inferenceTime));
        });
    }

    private String createOutputText(String prediction, float fps, long inferenceTime)
    {
        return String.format("%s\nFPS: %d\nInference time: %dms", prediction, (int)fps, (int)inferenceTime);
    }

    private Bitmap proxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    public void onRequestPermissionsResult (int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (arePermissionsGranted()) {
                startCamera();
            }
        }
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this, new String[] {CAMERA_PERMISSION}, PERMISSIONS_REQUEST_CODE);
    }

    private boolean arePermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED;
    }
}
