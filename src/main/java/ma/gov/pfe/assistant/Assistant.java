package ma.gov.pfe.assistant;

import dev.langchain4j.service.UserMessage;

public interface Assistant {

    @UserMessage("{{it}}")
    String chat(String userMessage);
}
