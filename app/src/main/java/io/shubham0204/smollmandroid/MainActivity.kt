/*
 * Copyright (C) 2024 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

package io.shubham0204.smollmandroid

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import io.shubham0204.smollmandroid.llm.ModelsRepository
import io.shubham0204.smollmandroid.ui.screens.chat.ChatActivity
import io.shubham0204.smollmandroid.ui.screens.model_download.DownloadModelActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.koin.android.ext.android.inject

class MainActivity : ComponentActivity() {
    private val modelsRepository by inject<ModelsRepository>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // A model is already registered -> go straight to chat.
        if (modelsRepository.getAvailableModelsList().isNotEmpty()) {
            goto(ChatActivity::class.java)
            return
        }

        // Gemma Pocket build: a GGUF ships inside the APK. On first launch we copy
        // it out and register it, then open chat. The copy of a ~1.7 GB file can
        // take up to a minute, so we do it off the main thread behind a splash.
        if (modelsRepository.hasBundledModel()) {
            setContent { SetupScreen() }
            lifecycleScope.launch(Dispatchers.IO) {
                modelsRepository.installBundledModel()
                withContext(Dispatchers.Main) { goto(ChatActivity::class.java) }
            }
            return
        }

        // Fallback (no bundled model): original behaviour, let the user download one.
        goto(DownloadModelActivity::class.java)
    }

    private fun goto(activity: Class<*>) {
        Intent(this, activity).apply {
            startActivity(this)
            finish()
        }
    }
}

@Composable
private fun SetupScreen() {
    Column(
        modifier = Modifier.fillMaxSize().background(Color(0xFF0B0F17)).padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator(color = Color(0xFF7C3AED))
        Text(
            text = "Preparing Gemma Pocket",
            color = Color.White,
            fontSize = 20.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(top = 24.dp),
        )
        Text(
            text = "First launch unpacks the on-device model. This can take a minute, and only happens once.",
            color = Color(0xFF94A3B8),
            fontSize = 14.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(top = 8.dp),
        )
    }
}
