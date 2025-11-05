package ma.gov.pfe.test5_websearch;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import ma.gov.pfe.assistant.Assistant;
import java.net.URL;
import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;

public class TestWebSearch {

    public static void main(String[] args) throws Exception {

        String apiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_KEY");

        if (apiKey == null || tavilyKey == null) {
            System.out.println(" Configure GEMINI_KEY et TAVILY_API_KEY avant d'exécuter !");
            return;
        }

        // Modèle Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // Embedding model + stockage du PDF
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> store = loadPdf("/rag.pdf", embeddingModel);
        if (store == null) {
            System.out.println("Impossible de charger le PDF /rag.pdf depuis resources.");
            return;
        }

        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();

        TavilyWebSearchEngine webEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webEngine)
                .maxResults(3)
                .build();

        // Router : va utiliser les deux retrievers (PDF + Web)
        var router = new DefaultQueryRouter(List.of(pdfRetriever, webRetriever));

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        System.out.println("Test 5 prêt — PDF + Recherche Web (exit pour quitter)");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print(" Question : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;

            System.out.println("\n Réponse : " + assistant.chat(question) + "\n");
        }
    }

    private static EmbeddingStore<TextSegment> loadPdf(String resourcePath, EmbeddingModel embeddingModel) throws Exception {
        URL resource = TestWebSearch.class.getResource(resourcePath);
        if (resource == null) {
            System.err.println(" Resource introuvable: " + resourcePath + " — vérifie que rag.pdf est dans src/main/resources");
            return null;
        }

        Document doc = FileSystemDocumentLoader.loadDocument(
                Path.of(resource.toURI()),
                new ApacheTikaDocumentParser()
        );

        var segments = DocumentSplitters.recursive(500, 50).split(doc);

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddingModel.embedAll(segments).content(), segments);
        return store;
    }
}
