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
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.os.Debug;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
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
    private CheckBox fp16Checkbox;
    private ImageView imageView;
    //private Classifier classifier;
    private Segmentator segmentator;
    private boolean reduceFp32ToFp16 = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = (PreviewView)findViewById(R.id.previewView);
        outputTextView = (TextView)findViewById(R.id.outputTextView);
        fp16Checkbox = (CheckBox)findViewById(R.id.fp16Checkbox);
        imageView = (ImageView)findViewById(R.id.imageView);
        //classifier = new Classifier(getAssets());
        segmentator = new Segmentator(getAssets(), reduceFp32ToFp16);

        if(arePermissionsGranted()) {
            previewView.post(() -> startCamera());
        }
        else {
            requestPermissions();
        }

        fp16Checkbox.setChecked(reduceFp32ToFp16);
        /*fp16Checkbox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener()
        {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                reduceFp32ToFp16 = isChecked;
                segmentator.destroy();
                segmentator = new Segmentator(getAssets(), reduceFp32ToFp16);
            }
        });*/
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
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setTargetResolution(analysisSize)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        //analysis.setAnalyzer(Executors.newSingleThreadExecutor(), image -> classify(image));
        analysis.setAnalyzer(Executors.newSingleThreadExecutor(), image -> doSegmentation(image));

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis);
        preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));
    }

    /*private void classify(ImageProxy image) {
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
    }*/

    private void doSegmentation(ImageProxy image) {
        if(reduceFp32ToFp16 != fp16Checkbox.isChecked()) {
            reduceFp32ToFp16 = fp16Checkbox.isChecked();
            segmentator.destroy();
            segmentator = new Segmentator(getAssets(), reduceFp32ToFp16);
        }

        long startTime = System.currentTimeMillis();

        Bitmap bitmap = proxyToBitmap(image);
        image.close();

        int rotation = image.getImageInfo().getRotationDegrees();
        if(rotation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(180 - rotation);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }

        float[] maskData = segmentator.predict(bitmap);
        bitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true);
        applyMask(bitmap, maskData);
        //Bitmap mask = maskToBitmap(maskData);

        long frameTime = System.currentTimeMillis() - startTime;
        float fps = 1000.0f / frameTime;

        long inferenceTime = segmentator.getLatestInferenceTime();

        Bitmap finalBitmap = bitmap;
        runOnUiThread(() -> {
            outputTextView.setText(createOutputText("", fps, inferenceTime));
            imageView.setImageBitmap(finalBitmap);
        });
    }

    private void applyMask(Bitmap frame, float[] mask) {
        for(int y = 0; y < frame.getHeight(); y++) {
            for(int x = 0; x < frame.getWidth(); x++) {
                float value = mask[y * frame.getWidth() + x];
                if(value < 0.5f) {
                    frame.setPixel(x, y, Color.BLACK);
                }
            }
        }
    }

    private Bitmap maskToBitmap(float[] mask) {
        Bitmap bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888);
        for(int y = 0; y < bitmap.getHeight(); y++) {
            for(int x = 0; x < bitmap.getWidth(); x++) {
                float value = mask[y * bitmap.getWidth() + x];
                bitmap.setPixel(x, y, getIntFromColor(value, value, value));
            }
        }
        return bitmap;
    }

    public int getIntFromColor(float r, float g, float b){
        int rb = Math.round(255 * r);
        int gb = Math.round(255 * g);
        int bb = Math.round(255 * b);

        rb = (rb << 16) & 0x00FF0000;
        gb = (gb << 8) & 0x0000FF00;
        bb = bb & 0x000000FF;

        return 0xFF000000 | rb | gb | bb;
    }

    private String createOutputText(String prediction, float fps, long inferenceTime) {
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
