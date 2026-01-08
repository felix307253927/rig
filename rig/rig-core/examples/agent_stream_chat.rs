use rig::agent::stream_to_stdout;
use rig::message::Message;
use rig::prelude::*;
use rig::streaming::StreamingChat;

use rig::providers;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv()?;
    tracing_subscriber::fmt().init();
    // Create OpenAI client
    let client = providers::openai::Client::from_env().completions_api();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(std::env::var("MODEL")?)
        .preamble("You are a comedian here to entertain the user using humour and jokes.用中文回答")
        .build();

    let messages = vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ];

    // Prompt the agent and print the response
    let mut stream = comedian_agent.stream_chat("Entertain me!", messages).await;

    let res = stream_to_stdout(&mut stream).await.unwrap();

    tracing::info!("Response: {res:?}");

    Ok(())
}
