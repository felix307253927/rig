use rig::prelude::*;
use rig::{
    Embed,
    completion::{CompletionModel, Prompt},
    embeddings::EmbeddingsBuilder,
    providers::openai,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::Serialize;
use std::vec;

// Data to be RAGged.
// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
#[derive(Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

#[derive(Clone, Default)]
struct RagHook;

impl<M: CompletionModel> rig::agent::PromptHook<M> for RagHook {
    async fn on_completion_call(
        &self,
        prompt: &rig::message::Message,
        _history: &[rig::message::Message],
        _cancel_sig: rig::agent::CancelSignal,
    ) {
        tracing::info!("[RAG] Prompt: {:?}", prompt);
    }
    async fn on_completion_response(
        &self,
        _prompt: &rig::message::Message,
        response: &rig::completion::CompletionResponse<M::Response>,
        _cancel_sig: rig::agent::CancelSignal,
    ) {
        tracing::info!("[RAG] Token: {:?}", response.usage);
        tracing::info!("[RAG] Choice: {:?}", response.choice);
        if let Ok(resp) = serde_json::to_value(&response.raw_response) {
            tracing::info!("[RAG] Received response: {}", resp);
        } else {
            tracing::error!("[RAG] Received response: <non-serializable>",);
        }
    }
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _args: &str,
        _cancel_sig: rig::agent::CancelSignal,
    ) {
        tracing::info!("[RAG] Tool call: {:?}", tool_name);
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _args: &str,
        _result: &str,
        _cancel_sig: rig::agent::CancelSignal,
    ) {
        tracing::info!("[RAG] Tool result: {:?}", tool_name);
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv()?;
    tracing_subscriber::fmt().init();

    // Create OpenAI client
    let openai_client = openai::Client::from_env();
    let embedding_model = openai_client
        .embedding_model(std::env::var("EMBEDDING_MODEL")?)
        .encoding_format(openai::EncodingFormat::Float);

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(vec![
            WordDefinition {
                id: "doc0".to_string(),
                word: "flurbo".to_string(),
                definitions: vec![
                    "1. *flurbo* (name): A flurbo is a green alien that lives on cold planets.".to_string(),
                    "2. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
                ]
            },
            WordDefinition {
                id: "doc1".to_string(),
                word: "glarb-glarb".to_string(),
                definitions: vec![
                    "1. *glarb-glarb* (noun): A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                    "2. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
                ]
            },
            WordDefinition {
                id: "doc2".to_string(),
                word: "linglingdong".to_string(),
                definitions: vec![
                    "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
                    "2. *linglingdong* (noun): A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
                ]
            },
        ])?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    // Create vector store index
    let index = vector_store.index(embedding_model);
    let rag_agent = openai_client.completions_api().agent(std::env::var("MODEL")?)
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.用中文回答
        ")
        .dynamic_context(1, index)
        .build();

    // Prompt the agent and print the response
    let prompt_request = rag_agent
        .prompt("What does \"glarb-glarb\" mean?")
        .with_hook(RagHook::default());
    let response = prompt_request.await?;

    tracing::info!("\nResponse:\n\n{response}");

    Ok(())
}
