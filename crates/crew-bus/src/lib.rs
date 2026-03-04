//! Message bus, channels, and session management for crew-rs gateway.

pub mod bus;
pub mod channel;
pub mod cli_channel;
pub mod coalesce;
pub mod cron_service;
pub mod cron_types;
pub mod heartbeat;
pub mod markdown_html;
pub mod media;
pub mod session;

#[cfg(feature = "discord")]
pub mod discord_channel;
#[cfg(feature = "email")]
pub mod email_channel;
#[cfg(feature = "feishu")]
pub mod feishu_channel;
#[cfg(feature = "slack")]
pub mod slack_channel;
#[cfg(feature = "telegram")]
pub mod telegram_channel;
#[cfg(feature = "twilio")]
pub mod twilio_channel;
#[cfg(feature = "wecom")]
pub mod wecom_channel;
#[cfg(feature = "whatsapp")]
pub mod whatsapp_channel;

pub use bus::{AgentHandle, BusPublisher, create_bus};
pub use channel::{Channel, ChannelManager};
pub use cli_channel::CliChannel;
pub use cron_service::CronService;
pub use cron_types::{CronJob, CronPayload, CronSchedule, CronStore};
pub use heartbeat::HeartbeatService;
pub use session::{
    ActiveSessionStore, Session, SessionListEntry, SessionManager, validate_topic_name,
};

#[cfg(feature = "discord")]
pub use discord_channel::DiscordChannel;
#[cfg(feature = "email")]
pub use email_channel::EmailChannel;
#[cfg(feature = "feishu")]
pub use feishu_channel::FeishuChannel;
#[cfg(feature = "slack")]
pub use slack_channel::SlackChannel;
#[cfg(feature = "telegram")]
pub use telegram_channel::TelegramChannel;
#[cfg(feature = "twilio")]
pub use twilio_channel::TwilioChannel;
#[cfg(feature = "wecom")]
pub use wecom_channel::WeComChannel;
#[cfg(feature = "whatsapp")]
pub use whatsapp_channel::WhatsAppChannel;
