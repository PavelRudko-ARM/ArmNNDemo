package com.arm.armnn_demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class Segmentator {

    public static final int INPUT_WIDTH = 128;
    public static final int INPUT_HEIGHT = 128;
    public static final int OUTPUT_CHANNELS = 1;

    private float[] maskData = new float[INPUT_WIDTH * INPUT_HEIGHT * OUTPUT_CHANNELS];
    private long latestInferenceTime = 0;

    public Segmentator(AssetManager assetManager, boolean reduceFp32ToFp16) {
        prepareNetwork(assetManager, reduceFp32ToFp16);
    }

    public float[] predict(Bitmap bitmap) {
        bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true);
        float[] imageData = getImageData(bitmap);
        long startTime = System.currentTimeMillis();
        runNetwork(imageData, maskData);
        latestInferenceTime = System.currentTimeMillis() - startTime;

        return maskData;
    }

    public long getLatestInferenceTime() {
        return latestInferenceTime;
    }

    public void destroy() {
        cleanupNetwork();
    }

    private float[] getImageData(Bitmap image) {
        int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

        float[] imageData = new float[pixels.length * 3];
        for(int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            int r = (color >> 16) & 0xff;
            int g = (color >> 8) & 0xff;
            int b = color & 0xff;

            imageData[i * 3] = (float)r / 255.0f;
            imageData[i * 3 + 1] = (float)g / 255.0f;
            imageData[i * 3 + 2] = (float)b / 255.0f;
        }

        return imageData;
    }

    private native void prepareNetwork(AssetManager assetManager, boolean reduceFp32ToFp16);
    private native void runNetwork(float[] imageData, float[] maskData);
    private native void cleanupNetwork();
}
