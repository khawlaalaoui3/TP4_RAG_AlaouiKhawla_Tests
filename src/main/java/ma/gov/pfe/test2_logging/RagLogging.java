package ma.gov.pfe.test2_logging;

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

public class RagLogging {

    public static void main(String[] args) throws Exception {

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) {
            System.out.println(" Définis GEMINI_KEY avant de lancer !");
            return;
        }

        // Logger Gemini activé
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)   // Logging activé
                .build();

        URL resource = RagLogging.class.getResource("/rag.pdf");
        Path path = Path.of(resource.toURI());

        var parser = new ApacheTikaDocumentParser();
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);

        var splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(doc);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

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

        Scanner sc = new Scanner(System.in);
        System.out.println("Test 2 Logging — Pose ta question ('exit' pour quitter)");

        while (true) {
            System.out.print(" Question: ");
            String q = sc.nextLine();
            if(q.equalsIgnoreCase("exit")) break;

            System.out.println(" " + assistant.chat(q));
        }
    }
}
