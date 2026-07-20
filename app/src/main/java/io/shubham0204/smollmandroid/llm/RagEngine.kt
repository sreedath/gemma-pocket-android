/*
 * Copyright (C) 2024 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

package io.shubham0204.smollmandroid.llm

import android.content.Context
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding
import org.json.JSONObject
import org.koin.core.annotation.Single
import java.io.File
import java.io.FileOutputStream

/**
 * On-device retrieval-augmented generation for Gemma Pocket. Everything is offline:
 * a MiniLM ONNX embedder (via the sentence-embeddings library) turns the question
 * into a vector, we cosine-match it against the bundled passage vectors, and the
 * top passages are wrapped into a RAFT-style context prompt for the model.
 *
 * The passage vectors in retrieval_pack.json were produced by the exact same MiniLM,
 * so query and passage embeddings are directly comparable (verified cosine 1.0).
 */
@Single
class RagEngine(private val context: Context) {
    private val embedder = SentenceEmbedding()
    private var passages: List<String> = emptyList()
    private var vectors: FloatArray = FloatArray(0) // flattened, n * dim
    private var dim = 384

    @Volatile
    private var ready = false

    companion object {
        private const val SYSTEM =
            "You are a helpful assistant. Answer the user's question using ONLY the " +
                "provided context. Some context passages are irrelevant. If the context " +
                "does not contain the answer, say so instead of guessing."
        private const val K = 3
        private const val ONNX = "minilm.onnx"
        private const val TOKENIZER = "minilm_tokenizer.json"
        private const val PACK = "retrieval_pack.json"
    }

    private fun hasAssets(): Boolean {
        val a = context.assets.list("") ?: return false
        return a.contains(ONNX) && a.contains(TOKENIZER) && a.contains(PACK)
    }

    /** Load the embedder and the passage vectors. Idempotent, one-time cost. */
    suspend fun init() {
        if (ready || !hasAssets()) return
        val onnx = copyAsset(ONNX)
        val tok = copyAsset(TOKENIZER)
        embedder.init(
            modelFilepath = onnx.absolutePath,
            tokenizerBytes = tok.readBytes(),
            useTokenTypeIds = false,
            outputTensorName = "sentence_embedding",
            useFP16 = false,
            useXNNPack = false,
        )
        val json = context.assets.open(PACK).bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        dim = obj.getInt("dim")
        val ps = obj.getJSONArray("passages")
        val es = obj.getJSONArray("embeddings")
        val n = ps.length()
        val list = ArrayList<String>(n)
        val flat = FloatArray(n * dim)
        for (i in 0 until n) {
            list.add(ps.getString(i))
            val row = es.getJSONArray(i)
            val off = i * dim
            for (j in 0 until dim) flat[off + j] = row.getDouble(j).toFloat()
        }
        passages = list
        vectors = flat
        ready = true
    }

    /**
     * Retrieve the top-K passages for [query] and wrap them into a RAFT context
     * prompt. Falls back to the bare query if retrieval is unavailable.
     */
    suspend fun buildRagPrompt(query: String): String {
        if (!ready) init()
        if (!ready || passages.isEmpty()) return query
        val q = embedder.encode(query)
        if (q.size != dim) return query
        val n = passages.size
        val scores = FloatArray(n)
        for (i in 0 until n) {
            var s = 0f
            val off = i * dim
            for (j in 0 until dim) s += vectors[off + j] * q[j]
            scores[i] = s
        }
        val top = (0 until n).sortedByDescending { scores[it] }.take(K)
        val ctx =
            top.mapIndexed { idx, pi -> "<context id=$idx>\n${passages[pi]}\n</context>" }
                .joinToString("\n\n")
        return "$SYSTEM\n\n$ctx\n\nQuestion: $query"
    }

    private fun copyAsset(name: String): File {
        val f = File(context.filesDir, name)
        if (!f.exists() || f.length() == 0L) {
            context.assets.open(name).use { input ->
                FileOutputStream(f).use { output -> input.copyTo(output) }
            }
        }
        return f
    }
}
