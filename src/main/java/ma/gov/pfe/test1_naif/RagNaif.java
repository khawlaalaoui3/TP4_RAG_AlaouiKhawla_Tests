package ma.gov.pfe.test1_naif;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import ma.gov.pfe.assistant.Assistant;

import java.net.URL;
import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) throws Exception {

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) {
            System.out.println("Définis GEMINI_KEY avant de lancer !");
            return;
        }

        //  Modèle LLM
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();

        //  Charger PDF depuis resources
        URL res = RagNaif.class.getResource("/rag.pdf");
        if (res == null) throw new RuntimeException(" Fichier rag.pdf introuvable !");
        Path path = Path.of(res.toURI());

        var parser = new ApacheTikaDocumentParser();
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);

        // Split du document
        var splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(doc);

        //  Embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // Stocker embeddings
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        // Retriever (RAG)
        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();

        // Questions
        Scanner sc = new Scanner(System.in);
        System.out.println(" RAG Naïf prêt — tape `exit` pour quitter");

        while (true) {
            System.out.print("Question : ");
            String q = sc.nextLine();
            if (q.equalsIgnoreCase("exit")) break;
            System.out.println(" " + assistant.chat(q));
        }
    }
}
