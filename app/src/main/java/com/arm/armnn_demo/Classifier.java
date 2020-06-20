package com.arm.armnn_demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.io.IOException;
import java.io.InputStream;

public class Classifier {

    private String[] classLabels;

    public Classifier(AssetManager assetManager) {
        prepareNetwork(assetManager);
        classLabels = loadText(assetManager, "imagenet_class_labels.txt").split("\\r?\\n");
    }

    public String predict(Bitmap bitmap) {
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        float[] imageData = getImageData(bitmap);
        int prediction = runNetwork(imageData);

        return classLabels[prediction];
    }

    public void destroy() {
        cleanupNetwork();
    }

    public String loadText(AssetManager assetManager, String path) {
        InputStream inputStream = null;
        String text = null;

        try {
            inputStream = assetManager.open(path);
            int size = inputStream.available();
            byte[] buffer = new byte[size];
            inputStream.read(buffer);
            inputStream.close();
            text = new String(buffer);
        } catch (IOException e) {
            text = null;
        }
        finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return text;
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

    private native void prepareNetwork(AssetManager assetManager);
    private native int runNetwork(float[] imageData);
    private native void cleanupNetwork();
}
