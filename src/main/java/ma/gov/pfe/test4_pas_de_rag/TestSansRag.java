package ma.gov.pfe.test4_pas_de_rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import ma.gov.pfe.assistant.Assistant;

import java.net.URL;
import java.nio.file.Path;
import java.util.*;

public class TestSansRag {

    public static void main(String[] args) throws Exception {

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) {
            System.out.println(" Définit GEMINI_KEY avant de lancer !");
            return;
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Charger le PDF RAG uniquement (assure-toi que /rag.pdf est dans src/main/resources)
        EmbeddingStore<TextSegment> store = loadPdf("/rag.pdf", embeddingModel);

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();

        // -------- ROUTER PERSONNALISÉ (prend un seul paramètre Query) --------
        QueryRouter router = (query) -> {
            // query.text() contient la question de l'utilisateur
            String userMessage = query.text();

            String template = """
                Est-ce que la requête suivante concerne l’intelligence artificielle ?
                Réponds seulement par : oui / non / peut-être

                Requête : {{question}}
            """;

            Prompt prompt = PromptTemplate.from(template)
                    .apply(Map.of("question", userMessage));

            // Convertir le Prompt en texte
            String promptText = prompt.text();

            // Appel modèle + nettoyage réponse (model.chat attend une String)
            String reply = model.chat(promptText)
                    .trim()
                    .toLowerCase();

            System.out.println("Décision IA ? → " + reply);

            if (reply.contains("non")) {
                return Collections.emptyList();
            } else {
                return List.of(retriever);
            }
        };

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println(" Test 4 prêt — écris une question (exit pour quitter)");

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print(" Question : ");
            String q = sc.nextLine();
            if (q.equalsIgnoreCase("exit")) break;

            System.out.println(" Réponse : " + assistant.chat(q));
        }
    }

    private static EmbeddingStore<TextSegment> loadPdf(String path, EmbeddingModel embeddingModel) throws Exception {
        URL resource = TestSansRag.class.getResource(path);
        if (resource == null) {
            throw new RuntimeException("Fichier ressource introuvable : " + path + " — place le dans src/main/resources");
        }
        Document doc = FileSystemDocumentLoader.loadDocument(Path.of(resource.toURI()), new ApacheTikaDocumentParser());

        var splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(doc);

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddingModel.embedAll(segments).content(), segments);
        return store;
    }
}
