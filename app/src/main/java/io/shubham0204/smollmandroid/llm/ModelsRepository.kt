/*
 * Copyright (C) 2024 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

package io.shubham0204.smollmandroid.llm

import android.content.Context
import io.shubham0204.smollm.GGUFReader
import io.shubham0204.smollm.SmolLM
import io.shubham0204.smollmandroid.data.AppDB
import io.shubham0204.smollmandroid.data.LLMModel
import io.shubham0204.smollmandroid.ui.screens.manage_asr.ASRModel
import kotlinx.coroutines.flow.Flow
import org.koin.core.annotation.Single
import java.io.File
import java.io.FileOutputStream

@Single
class ModelsRepository(private val context: Context, private val appDB: AppDB) {
    init {
        for (model in appDB.getModelsList()) {
            if (!File(model.path).exists()) {
                deleteModel(model.id)
            }
        }
    }

    companion object {
        // Gemma Pocket: the model that ships bundled inside the APK.
        const val BUNDLED_ASSET = "gemma-2-2b-qa-raft-Q4_K_M.gguf"
        const val BUNDLED_NAME = "Gemma 2B RAFT (bundled)"

        fun checkIfModelsDownloaded(context: Context): Boolean {
            context.filesDir.listFiles()?.forEach { file ->
                if (file.isFile && file.name.endsWith(".gguf")) {
                    return true
                }
            }
            return false
        }
    }

    /** True if a GGUF is bundled in the APK assets (Gemma Pocket build). */
    fun hasBundledModel(): Boolean =
        (context.assets.list("")?.contains(BUNDLED_ASSET)) == true

    /**
     * First launch of the bundled build: copy the GGUF out of assets into
     * filesDir (llama.cpp needs a real file path) and register it, reading the
     * context size + chat template straight from the GGUF, exactly like import.
     * Safe to call repeatedly; it no-ops once a model is registered.
     */
    fun installBundledModel() {
        if (appDB.getModelsList().isNotEmpty()) return
        if (!hasBundledModel()) return
        val dest = File(context.filesDir, BUNDLED_ASSET)
        if (!dest.exists() || dest.length() == 0L) {
            context.assets.open(BUNDLED_ASSET).use { input ->
                FileOutputStream(dest).use { output -> input.copyTo(output) }
            }
        }
        val ggufReader = GGUFReader()
        ggufReader.load(dest.absolutePath)
        val contextSize =
            ggufReader.getContextSize() ?: SmolLM.DefaultInferenceParams.contextSize
        val chatTemplate =
            ggufReader.getChatTemplate() ?: SmolLM.DefaultInferenceParams.chatTemplate
        appDB.addModel(BUNDLED_NAME, "", dest.absolutePath, contextSize.toInt(), chatTemplate)
    }

    fun getModelFromId(id: Long): LLMModel = appDB.getModel(id)

    fun getAvailableModels(): Flow<List<LLMModel>> = appDB.getModels()

    fun getAvailableModelsList(): List<LLMModel> = appDB.getModelsList()

    fun deleteModel(id: Long) {
        appDB.getModel(id).also {
            File(it.path).delete()
            appDB.deleteModel(it.id)
        }
    }

    fun isSpeech2TextModelDownloaded(asrModel: ASRModel): Boolean {
        return File(context.filesDir, asrModel.name).exists()
    }
}
