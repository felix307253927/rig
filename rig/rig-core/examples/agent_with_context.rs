use rig::prelude::*;
use rig::{agent::AgentBuilder, completion::Prompt, providers::openai};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv()?;
    tracing_subscriber::fmt().init();
    // Create OpenAI client
    let client: rig::client::Client<openai::OpenAICompletionsExt> =
        openai::Client::from_env().completions_api();

    let model = client.completion_model(std::env::var("MODEL")?);

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .context("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .preamble("用中文回答")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    tracing::info!("{response}");

    Ok(())
}
