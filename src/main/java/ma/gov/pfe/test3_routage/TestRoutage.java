package ma.gov.pfe.test3_routage;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import ma.gov.pfe.assistant.Assistant;
import java.net.URL;
import java.nio.file.Path;
import java.util.*;

public class TestRoutage {

    public static void main(String[] args) throws Exception {

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) {
            System.out.println(" Définis GEMINI_KEY avant de lancer !");
            return;
        }

        //  Modèle Gemini avec logs
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        //  Charger deux documents
        EmbeddingStore<TextSegment> storeIA = loadDocument("/rag.pdf");
        EmbeddingStore<TextSegment> storeAutre = loadDocument("/langchain4jj.pdf");

        //  Modèle embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverAutre = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeAutre)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        //  Routage via LLM
        Map<ContentRetriever, String> routeMap = new HashMap<>();
        routeMap.put(retrieverIA, "Document sur le RAG, IA, embeddings, retrieval.");
        routeMap.put(retrieverAutre, "Document sur un sujet non-IA (général).");

        QueryRouter router = new LanguageModelQueryRouter(model, routeMap);

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        Scanner sc = new Scanner(System.in);
        System.out.println(" Routage prêt — pose une question ('exit' pour quitter)");

        while (true) {
            System.out.print(" Question : ");
            String q = sc.nextLine();

            if (q.equalsIgnoreCase("exit")) break;
            System.out.println("🤖 " + assistant.chat(q));
        }
    }

    //  Charger document PDF et embeddings
    private static EmbeddingStore<TextSegment> loadDocument(String resourcePath) throws Exception {

        URL resource = TestRoutage.class.getResource(resourcePath);
        if (resource == null) {
            throw new RuntimeException(" Fichier introuvable dans resources: " + resourcePath);
        }

        System.out.println(" Chargement du fichier : " + resourcePath);

        Path path = Path.of(resource.toURI());
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());

        var splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(doc);

        EmbeddingModel embModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        return store;
    }
}
