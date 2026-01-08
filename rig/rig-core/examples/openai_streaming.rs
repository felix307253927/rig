use rig::agent::stream_to_stdout;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv()?;
    // Uncomment tracing for debugging
    tracing_subscriber::fmt().init();

    // Create streaming agent with a single context prompt
    let agent = openai::Client::from_env()
        .completions_api()
        .agent(std::env::var("MODEL")?)
        .preamble("Be precise and concise.用中文回答")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;

    let res = stream_to_stdout(&mut stream).await?;

    tracing::info!("Token usage response: {usage:?}", usage = res.usage());
    tracing::info!("Final text response: {message:?}", message = res.response());

    Ok(())
}
