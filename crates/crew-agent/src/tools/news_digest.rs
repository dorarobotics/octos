//! News digest tool: hybrid mode — discovers headlines from RSS/API sources
//! (Google News, Hacker News, Yahoo, Substack, Medium), then deep-fetches
//! top articles for full content, and synthesizes a structured digest via LLM.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use crew_core::{Message, MessageRole, TokenUsage};
use crew_llm::{ChatConfig, LlmProvider};
use eyre::{Result, WrapErr};
use serde::Deserialize;
use tracing::{info, warn};

use super::{Tool, ToolResult};

// ---------------------------------------------------------------------------
// Source types
// ---------------------------------------------------------------------------

/// How to fetch a particular news source.
enum SourceKind {
    /// Google News RSS — titles have "Headline - Source" format, links are Google redirects.
    GoogleRss(&'static str),
    /// Generic RSS/Atom — Substack, Medium, etc.
    GenericRss(&'static str),
    /// Yahoo News HTML — fetch + htmd (no article URL extraction).
    YahooHtml(&'static str),
    /// Hacker News Firebase API.
    HackerNewsApi,
}

struct SourceDef {
    name: &'static str,
    kind: SourceKind,
}

/// Result from fetching one source: headline text + discovered article URLs.
struct FetchResult {
    /// Formatted headlines text for LLM context.
    text: String,
    /// Article URLs discovered (title, url) for deep-fetch phase.
    article_urls: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Category definitions
// ---------------------------------------------------------------------------

struct CategoryDef {
    name: &'static str,
    label_zh: &'static str,
    sources: &'static [SourceDef],
    /// Max articles to deep-fetch from this category.
    deep_fetch_limit: usize,
}

macro_rules! sources {
    ($($def:expr),+ $(,)?) => { &[$($def),+] };
}

const CATEGORIES: &[CategoryDef] = &[
    CategoryDef {
        name: "politics",
        label_zh: "美国政治",
        deep_fetch_limit: 3,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNRGxqTjNjd0VnSmxiaWdBUAE?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/politics/")
            },
        ],
    },
    CategoryDef {
        name: "world",
        label_zh: "国际新闻",
        deep_fetch_limit: 3,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/world/")
            },
        ],
    },
    CategoryDef {
        name: "business",
        label_zh: "商业财经",
        deep_fetch_limit: 3,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/business/")
            },
        ],
    },
    CategoryDef {
        name: "technology",
        label_zh: "科技",
        deep_fetch_limit: 10,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Hacker News",
                kind: SourceKind::HackerNewsApi
            },
            SourceDef {
                name: "Substack",
                kind: SourceKind::GenericRss("https://newsletter.pragmaticengineer.com/feed")
            },
            SourceDef {
                name: "Substack (AI)",
                kind: SourceKind::GenericRss("https://www.oneusefulthing.org/feed")
            },
            SourceDef {
                name: "Medium",
                kind: SourceKind::GenericRss("https://medium.com/feed/tag/technology")
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/technology/")
            },
        ],
    },
    CategoryDef {
        name: "science",
        label_zh: "科学",
        deep_fetch_limit: 3,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/science/")
            },
        ],
    },
    CategoryDef {
        name: "entertainment",
        label_zh: "社会娱乐",
        deep_fetch_limit: 2,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/entertainment/")
            },
        ],
    },
    CategoryDef {
        name: "health",
        label_zh: "健康",
        deep_fetch_limit: 2,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo News",
                kind: SourceKind::YahooHtml("https://news.yahoo.com/health/")
            },
        ],
    },
    CategoryDef {
        name: "sports",
        label_zh: "体育",
        deep_fetch_limit: 2,
        sources: sources![
            SourceDef {
                name: "Google News",
                kind: SourceKind::GoogleRss(
                    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
                )
            },
            SourceDef {
                name: "Yahoo Sports",
                kind: SourceKind::YahooHtml("https://sports.yahoo.com/")
            },
        ],
    },
];

const ALIASES: &[(&str, &str)] = &[
    ("tech", "technology"),
    ("commerce", "business"),
    ("international", "world"),
    ("social", "entertainment"),
];

/// Max chars per HTML source page.
const MAX_SOURCE_CHARS: usize = 12_000;
/// Max chars per deep-fetched article.
const MAX_ARTICLE_CHARS: usize = 8_000;
/// Max HN stories to fetch in discovery phase.
const HN_TOP_STORIES: usize = 30;
/// Max RSS items to parse per feed.
const MAX_RSS_ITEMS: usize = 30;
/// Global cap on total deep-fetch articles (across all categories).
const MAX_DEEP_FETCH_TOTAL: usize = 20;

fn resolve_alias(name: &str) -> &str {
    let lower = name.to_lowercase();
    for &(alias, canonical) in ALIASES {
        if lower == alias {
            return canonical;
        }
    }
    for cat in CATEGORIES {
        if lower == cat.name {
            return cat.name;
        }
    }
    ""
}

fn find_category(name: &str) -> Option<&'static CategoryDef> {
    let canonical = resolve_alias(name);
    CATEGORIES.iter().find(|c| c.name == canonical)
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

pub struct NewsDigestTool {
    llm: Arc<dyn LlmProvider>,
    data_dir: PathBuf,
    client: reqwest::Client,
    config: Option<Arc<super::tool_config::ToolConfigStore>>,
}

impl NewsDigestTool {
    pub fn new(llm: Arc<dyn LlmProvider>, data_dir: impl Into<PathBuf>) -> Self {
        Self {
            llm,
            data_dir: data_dir.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::limited(5))
                .user_agent(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
                     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                )
                .build()
                .expect("failed to build HTTP client"),
            config: None,
        }
    }

    pub fn with_config(mut self, config: Arc<super::tool_config::ToolConfigStore>) -> Self {
        self.config = Some(config);
        self
    }

    /// Resolve configurable settings: per-call args > user config > hardcoded defaults.
    async fn resolve_settings(&self, input: &Input) -> ResolvedSettings {
        let (cfg_lang, cfg_hn, cfg_rss, cfg_deep, cfg_source_chars, cfg_article_chars) =
            match &self.config {
                Some(c) => (
                    c.get_str("news_digest", "language").await,
                    c.get_usize("news_digest", "hn_top_stories").await,
                    c.get_usize("news_digest", "max_rss_items").await,
                    c.get_usize("news_digest", "max_deep_fetch_total").await,
                    c.get_usize("news_digest", "max_source_chars").await,
                    c.get_usize("news_digest", "max_article_chars").await,
                ),
                None => (None, None, None, None, None, None),
            };

        ResolvedSettings {
            language: input
                .language
                .clone()
                .or(cfg_lang)
                .unwrap_or_else(|| "zh".into()),
            hn_top_stories: cfg_hn.unwrap_or(HN_TOP_STORIES),
            max_rss_items: cfg_rss.unwrap_or(MAX_RSS_ITEMS),
            max_deep_fetch_total: cfg_deep.unwrap_or(MAX_DEEP_FETCH_TOTAL),
            max_source_chars: cfg_source_chars.unwrap_or(MAX_SOURCE_CHARS),
            max_article_chars: cfg_article_chars.unwrap_or(MAX_ARTICLE_CHARS),
        }
    }

    // ---- Phase 1: Discovery (headlines + article URLs) ----

    async fn fetch_source(
        &self,
        source: &SourceDef,
        settings: &ResolvedSettings,
    ) -> Result<FetchResult> {
        match &source.kind {
            SourceKind::GoogleRss(url) => self.fetch_google_rss(url, settings.max_rss_items).await,
            SourceKind::GenericRss(url) => {
                self.fetch_generic_rss(url, settings.max_rss_items).await
            }
            SourceKind::YahooHtml(url) => {
                self.fetch_html_page(url, settings.max_source_chars).await
            }
            SourceKind::HackerNewsApi => self.fetch_hackernews(settings.hn_top_stories).await,
        }
    }

    /// Google News RSS — titles are "Headline - Source", links are redirect URLs.
    async fn fetch_google_rss(&self, url: &str, max_rss_items: usize) -> Result<FetchResult> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .wrap_err("RSS fetch failed")?;
        if !response.status().is_success() {
            eyre::bail!("RSS HTTP {}", response.status());
        }
        let xml = response.text().await.wrap_err("failed to read RSS body")?;

        let mut text = String::new();
        let mut urls = Vec::new();
        for (i, chunk) in xml.split("<item>").skip(1).enumerate() {
            if i >= max_rss_items {
                break;
            }
            if let Some(title) = extract_xml_tag(chunk, "title") {
                let title = decode_xml_entities(&title);
                text.push_str(&format!("{}. {}\n", i + 1, title));
                if let Some(link) = extract_xml_tag(chunk, "link") {
                    let link = decode_xml_entities(&link);
                    urls.push((title, link));
                }
            }
        }
        if text.is_empty() {
            eyre::bail!("no items found in RSS");
        }
        Ok(FetchResult {
            text,
            article_urls: urls,
        })
    }

    /// Generic RSS/Atom — for Substack, Medium, etc.
    async fn fetch_generic_rss(&self, url: &str, max_rss_items: usize) -> Result<FetchResult> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .wrap_err("RSS fetch failed")?;
        if !response.status().is_success() {
            eyre::bail!("RSS HTTP {}", response.status());
        }
        let xml = response.text().await.wrap_err("failed to read RSS body")?;

        let mut text = String::new();
        let mut urls = Vec::new();
        let mut i = 0;

        // Try RSS <item> format first, then Atom <entry> format
        let (item_split, link_tag) = if xml.contains("<item>") {
            ("<item>", "link")
        } else if xml.contains("<entry>") {
            ("<entry>", "id") // Atom uses <id> for URL, or <link href="..."/>
        } else {
            eyre::bail!("unrecognized feed format");
        };

        for chunk in xml.split(item_split).skip(1) {
            if i >= max_rss_items {
                break;
            }
            if let Some(title) = extract_xml_tag(chunk, "title") {
                let title = decode_xml_entities(&title);
                // Try to extract description/summary for richer context
                let desc = extract_xml_tag(chunk, "description")
                    .or_else(|| extract_xml_tag(chunk, "summary"))
                    .map(|d| {
                        let d = decode_xml_entities(&d);
                        // Strip HTML from description
                        extract_text_fallback(&d)
                    });

                text.push_str(&format!("{}. {}", i + 1, title));
                if let Some(ref desc) = desc {
                    let short: String = desc.chars().take(200).collect();
                    text.push_str(&format!(" — {short}"));
                }
                text.push('\n');

                // Extract link
                let link =
                    extract_xml_tag(chunk, link_tag).or_else(|| extract_atom_link_href(chunk));
                if let Some(link) = link {
                    let link = decode_xml_entities(&link);
                    urls.push((title, link));
                }
                i += 1;
            }
        }
        if text.is_empty() {
            eyre::bail!("no items found in feed");
        }
        Ok(FetchResult {
            text,
            article_urls: urls,
        })
    }

    /// Hacker News API — structured data with scores.
    async fn fetch_hackernews(&self, hn_top_stories: usize) -> Result<FetchResult> {
        let ids: Vec<u64> = self
            .client
            .get("https://hacker-news.firebaseio.com/v0/topstories.json")
            .send()
            .await
            .wrap_err("HN API failed")?
            .json()
            .await
            .wrap_err("HN JSON parse failed")?;

        let top_ids = &ids[..ids.len().min(hn_top_stories)];

        let fetches: Vec<_> = top_ids
            .iter()
            .map(|&id| async move {
                let url = format!("https://hacker-news.firebaseio.com/v0/item/{id}.json");
                self.client
                    .get(&url)
                    .send()
                    .await
                    .ok()?
                    .json::<serde_json::Value>()
                    .await
                    .ok()
            })
            .collect();

        let results = futures::future::join_all(fetches).await;

        let mut text = String::new();
        let mut urls = Vec::new();
        for (i, item) in results.into_iter().flatten().enumerate() {
            let title = item["title"].as_str().unwrap_or("(untitled)").to_string();
            let score = item["score"].as_u64().unwrap_or(0);
            let url = item["url"].as_str().unwrap_or("").to_string();
            let descendants = item["descendants"].as_u64().unwrap_or(0);

            text.push_str(&format!(
                "{}. [{}pts, {}comments] {}",
                i + 1,
                score,
                descendants,
                title
            ));
            if !url.is_empty() {
                text.push_str(&format!(" ({})", url));
                urls.push((title.clone(), url));
            }
            text.push('\n');
        }
        // Sort URLs by score (highest first) — they're already in order from API
        Ok(FetchResult {
            text,
            article_urls: urls,
        })
    }

    /// Yahoo/HTML page — text only, no URL extraction.
    async fn fetch_html_page(&self, url: &str, max_source_chars: usize) -> Result<FetchResult> {
        let response = self
            .client
            .get(url)
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await
            .wrap_err("HTTP request failed")?;

        if !response.status().is_success() {
            eyre::bail!("HTTP {}", response.status());
        }

        let html = response.text().await.wrap_err("failed to read body")?;
        let cleaned = strip_scripts(&html);
        let mut text = htmd::convert(&cleaned).unwrap_or_else(|_| extract_text_fallback(&cleaned));
        crew_core::truncate_utf8(&mut text, max_source_chars, "\n...(truncated)");
        Ok(FetchResult {
            text,
            article_urls: vec![],
        })
    }

    // ---- Phase 1.5: Fetch all sources for a category ----

    async fn fetch_category(
        &self,
        cat: &'static CategoryDef,
        settings: &ResolvedSettings,
    ) -> (String, String, Vec<(String, String)>) {
        // (label_zh, combined_text, collected_article_urls)
        let fetches: Vec<_> = cat
            .sources
            .iter()
            .map(|source| async move {
                match self.fetch_source(source, settings).await {
                    Ok(result) => {
                        info!(
                            "fetched {}/{}: {} chars, {} URLs",
                            cat.name,
                            source.name,
                            result.text.len(),
                            result.article_urls.len()
                        );
                        Some((source.name, result))
                    }
                    Err(e) => {
                        warn!("failed to fetch {}/{}: {e}", cat.name, source.name);
                        None
                    }
                }
            })
            .collect();

        let results = futures::future::join_all(fetches).await;

        let mut combined = String::new();
        let mut all_urls = Vec::new();
        for (source_name, result) in results.into_iter().flatten() {
            combined.push_str(&format!("--- {source_name} ---\n{}\n\n", result.text));
            all_urls.extend(result.article_urls);
        }

        (cat.label_zh.to_string(), combined, all_urls)
    }

    // ---- Phase 2: Deep fetch top articles ----

    /// Fetch a single article URL and return its content.
    async fn deep_fetch_article(
        &self,
        title: &str,
        url: &str,
        max_article_chars: usize,
    ) -> Option<String> {
        let response = self
            .client
            .get(url)
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        let html = response.text().await.ok()?;
        let cleaned = strip_scripts(&html);
        let mut text = htmd::convert(&cleaned).unwrap_or_else(|_| extract_text_fallback(&cleaned));
        crew_core::truncate_utf8(&mut text, max_article_chars, "\n...(truncated)");

        // Skip if too short (likely a paywall or redirect)
        if text.len() < 200 {
            return None;
        }

        Some(format!("### {title}\n_Source: {url}_\n\n{text}"))
    }

    // ---- Phase 3: Synthesis ----

    async fn synthesize_digest(
        &self,
        headlines: &[(String, String)], // (category_label, headlines_text)
        deep_content: &str,             // full article content
        language: &str,
    ) -> Result<(String, TokenUsage)> {
        let today = Utc::now().format("%Y-%m-%d").to_string();

        let mut content = String::new();

        // Headlines section
        content.push_str("## HEADLINES BY CATEGORY\n\n");
        for (label, text) in headlines {
            content.push_str(&format!("=== {label} ===\n{text}\n\n"));
        }

        // Deep content section
        if !deep_content.is_empty() {
            content.push_str("## FULL ARTICLE CONTENT (top stories)\n\n");
            content.push_str(deep_content);
        }

        let lang_instruction = if language == "en" {
            "Write the digest in English."
        } else {
            "Write the digest in Chinese (中文). Translate all headlines and summaries to Chinese."
        };

        let prompt = format!(
            "You are a professional news editor. Synthesize the following content into a \
             well-structured daily news digest for {today}.\n\n\
             You have two sections of input:\n\
             1. HEADLINES BY CATEGORY — headlines from Google News RSS, Hacker News, \
                Substack, Medium, and Yahoo News\n\
             2. FULL ARTICLE CONTENT — deep-fetched content for the most important stories\n\n\
             Instructions:\n\
             - {lang_instruction}\n\
             - Group by category with clear headers\n\
             - For stories with full article content, write detailed 2-3 sentence summaries\n\
             - For headline-only stories, write 1 sentence summaries based on the headline\n\
             - Include source name for each story\n\
             - Deduplicate stories that appear across multiple sources\n\
             - Skip ads, navigation text, and cookie notices\n\
             - Include 5-10 most important stories per category\n\
             - For Hacker News items, prioritize high-score stories\n\
             - Use markdown formatting\n\
             - Start with a title: \"# 每日新闻速递 {today}\" (or English equivalent)\n\n\
             Raw content:\n\n{content}"
        );

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            timestamp: Utc::now(),
        }];

        let config = ChatConfig {
            max_tokens: Some(8192),
            temperature: Some(0.0),
            ..Default::default()
        };

        let response = self.llm.chat(&messages, &[], &config).await?;
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        };
        Ok((response.content.unwrap_or_default(), usage))
    }
}

// ---------------------------------------------------------------------------
// Tool trait impl
// ---------------------------------------------------------------------------

/// Resolved settings after applying priority chain.
struct ResolvedSettings {
    language: String,
    hn_top_stories: usize,
    max_rss_items: usize,
    max_deep_fetch_total: usize,
    max_source_chars: usize,
    max_article_chars: usize,
}

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    categories: Vec<String>,
    #[serde(default)]
    language: Option<String>,
}

#[async_trait]
impl Tool for NewsDigestTool {
    fn name(&self) -> &str {
        "news_digest"
    }

    fn description(&self) -> &str {
        "Fetch news from Google News, Hacker News, Substack, Medium, and Yahoo News. \
         Uses hybrid mode: discovers headlines via RSS/API, then deep-fetches top articles \
         for full content, and synthesizes a structured digest. Categories: politics, \
         world/international, business/commerce, technology/tech, science, \
         entertainment/social, health, sports. Default language: Chinese."
    }

    fn tags(&self) -> &[&str] {
        &["web", "gateway"]
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "News categories to fetch. Empty array = all categories. \
                        Options: politics, world/international, business/commerce, \
                        technology/tech, science, entertainment/social, health, sports"
                },
                "language": {
                    "type": "string",
                    "enum": ["zh", "en"],
                    "description": "Output language: 'zh' (Chinese, default) or 'en' (English)"
                }
            }
        })
    }

    async fn execute(&self, args: &serde_json::Value) -> Result<ToolResult> {
        let input: Input =
            serde_json::from_value(args.clone()).wrap_err("invalid news_digest input")?;
        let settings = self.resolve_settings(&input).await;

        let targets: Vec<&CategoryDef> = if input.categories.is_empty() {
            CATEGORIES.iter().collect()
        } else {
            let mut resolved: Vec<&CategoryDef> = Vec::new();
            for name in &input.categories {
                if let Some(cat) = find_category(name) {
                    if !resolved.iter().any(|c| c.name == cat.name) {
                        resolved.push(cat);
                    }
                } else {
                    warn!("unknown news category: {name}");
                }
            }
            if resolved.is_empty() {
                return Ok(ToolResult {
                    output: format!(
                        "No valid categories found. Available: {}",
                        CATEGORIES
                            .iter()
                            .map(|c| c.name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    success: false,
                    ..Default::default()
                });
            }
            resolved
        };

        // ---- Phase 1: Discover headlines ----
        let total_sources: usize = targets.iter().map(|c| c.sources.len()).sum();
        info!(
            "Phase 1: discovering headlines from {} categories ({} sources)",
            targets.len(),
            total_sources
        );

        let fetches: Vec<_> = targets
            .iter()
            .map(|cat| self.fetch_category(cat, &settings))
            .collect();
        let results = futures::future::join_all(fetches).await;

        let mut headlines: Vec<(String, String)> = Vec::new();
        let mut category_urls: Vec<(&CategoryDef, Vec<(String, String)>)> = Vec::new();

        for (cat, (label, text, urls)) in targets.iter().zip(results.into_iter()) {
            if !text.is_empty() {
                headlines.push((label, text));
                if !urls.is_empty() {
                    category_urls.push((cat, urls));
                }
            }
        }

        if headlines.is_empty() {
            return Ok(ToolResult {
                output: "Failed to fetch any news sources.".to_string(),
                success: false,
                ..Default::default()
            });
        }

        // ---- Phase 2: Deep fetch top articles ----
        let mut deep_fetch_targets: Vec<(String, String)> = Vec::new();
        let mut seen_urls: HashSet<String> = HashSet::new();

        for (cat, urls) in &category_urls {
            let limit = cat.deep_fetch_limit;
            let mut added = 0;
            for (title, url) in urls.iter() {
                if added >= limit {
                    break;
                }
                // Skip Google News redirect URLs — they return a JS-rendered
                // intermediate page, not the actual article content.
                if url.contains("news.google.com/") {
                    continue;
                }
                // Skip duplicates
                if seen_urls.contains(url.as_str()) {
                    continue;
                }
                // Skip non-http URLs
                if !url.starts_with("http") {
                    continue;
                }
                seen_urls.insert(url.clone());
                deep_fetch_targets.push((title.clone(), url.clone()));
                added += 1;
                if deep_fetch_targets.len() >= settings.max_deep_fetch_total {
                    break;
                }
            }
            if deep_fetch_targets.len() >= settings.max_deep_fetch_total {
                break;
            }
        }

        let deep_content = if !deep_fetch_targets.is_empty() {
            for (i, (title, url)) in deep_fetch_targets.iter().enumerate() {
                info!(
                    "  deep-fetch [{}/{}]: {} — {}",
                    i + 1,
                    deep_fetch_targets.len(),
                    title,
                    url
                );
            }
            info!(
                "Phase 2: deep-fetching {} articles",
                deep_fetch_targets.len()
            );

            let max_article_chars = settings.max_article_chars;
            let fetches: Vec<_> = deep_fetch_targets
                .iter()
                .map(|(title, url)| self.deep_fetch_article(title, url, max_article_chars))
                .collect();

            let results = futures::future::join_all(fetches).await;
            let articles: Vec<String> = results.into_iter().flatten().collect();
            info!("Phase 2: got {} articles with content", articles.len());
            articles.join("\n\n---\n\n")
        } else {
            String::new()
        };

        // ---- Phase 3: Synthesize ----
        info!(
            "Phase 3: synthesizing digest from {} categories + {} deep articles",
            headlines.len(),
            deep_fetch_targets.len()
        );

        let (digest, usage) = self
            .synthesize_digest(&headlines, &deep_content, &settings.language)
            .await
            .wrap_err("LLM synthesis failed")?;

        // Save to disk
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let research_dir = self.data_dir.join("research");
        tokio::fs::create_dir_all(&research_dir).await.ok();
        let file_path = research_dir.join(format!("news-digest-{today}.md"));
        tokio::fs::write(&file_path, &digest).await.ok();
        info!("saved digest to {}", file_path.display());

        Ok(ToolResult {
            output: digest,
            success: true,
            tokens_used: Some(usage),
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract content of an XML tag (non-nested, first occurrence).
fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].to_string())
}

/// Extract href from Atom-style `<link href="..." />` or `<link rel="alternate" href="..."/>`.
fn extract_atom_link_href(xml: &str) -> Option<String> {
    // Look for href="..." in a <link> tag
    let link_start = xml.find("<link")?;
    let chunk = &xml[link_start..];
    let tag_end = chunk.find('>')?;
    let tag = &chunk[..tag_end];
    let href_start = tag.find("href=\"")? + 6;
    let href_end = tag[href_start..].find('"')? + href_start;
    Some(tag[href_start..href_end].to_string())
}

/// Decode common XML entities and strip CDATA wrappers.
fn decode_xml_entities(s: &str) -> String {
    let s = s.trim();
    // Strip CDATA wrapper: <![CDATA[...]]>
    let s = s
        .strip_prefix("<![CDATA[")
        .and_then(|inner| inner.strip_suffix("]]>"))
        .unwrap_or(s);
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

/// Strip `<script>` and `<style>` tags and their content from HTML.
fn strip_scripts(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let lower = html.to_lowercase();
    let bytes = html.as_bytes();
    let lower_bytes = lower.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if i + 7 < lower_bytes.len() && &lower_bytes[i..i + 7] == b"<script" {
            if let Some(end) = lower[i..].find("</script>") {
                i += end + 9;
                continue;
            }
        }
        if i + 6 < lower_bytes.len() && &lower_bytes[i..i + 6] == b"<style" {
            if let Some(end) = lower[i..].find("</style>") {
                i += end + 8;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Simple HTML tag stripper as fallback when htmd fails.
fn extract_text_fallback(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        if c == '<' {
            in_tag = true;
        } else if c == '>' {
            in_tag = false;
            result.push(' ');
        } else if !in_tag {
            result.push(c);
        }
    }
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_alias() {
        assert_eq!(resolve_alias("tech"), "technology");
        assert_eq!(resolve_alias("Tech"), "technology");
        assert_eq!(resolve_alias("commerce"), "business");
        assert_eq!(resolve_alias("international"), "world");
        assert_eq!(resolve_alias("social"), "entertainment");
        assert_eq!(resolve_alias("politics"), "politics");
        assert_eq!(resolve_alias("unknown"), "");
    }

    #[test]
    fn test_find_category() {
        let cat = find_category("tech").unwrap();
        assert_eq!(cat.name, "technology");
        assert_eq!(cat.label_zh, "科技");
        assert!(cat.sources.len() >= 4); // Google + HN + Substack + Medium + Yahoo

        let cat = find_category("international").unwrap();
        assert_eq!(cat.name, "world");
        assert_eq!(cat.label_zh, "国际新闻");

        assert!(find_category("nonexistent").is_none());
    }

    #[test]
    fn test_all_categories_have_sources() {
        for cat in CATEGORIES {
            assert!(!cat.name.is_empty());
            assert!(!cat.label_zh.is_empty());
            assert!(!cat.sources.is_empty(), "{} has no sources", cat.name);
            assert!(
                cat.deep_fetch_limit > 0,
                "{} has 0 deep_fetch_limit",
                cat.name
            );
        }
    }

    #[test]
    fn test_extract_xml_tag() {
        let xml = "<title>Hello World - CNN</title><link>http://example.com</link>";
        assert_eq!(
            extract_xml_tag(xml, "title"),
            Some("Hello World - CNN".to_string())
        );
        assert_eq!(
            extract_xml_tag(xml, "link"),
            Some("http://example.com".to_string())
        );
        assert_eq!(extract_xml_tag(xml, "missing"), None);
    }

    #[test]
    fn test_extract_atom_link_href() {
        let xml = r#"<link rel="alternate" href="https://example.com/post"/>"#;
        assert_eq!(
            extract_atom_link_href(xml),
            Some("https://example.com/post".to_string())
        );

        let xml = r#"<link href="https://sub.example.com/feed" />"#;
        assert_eq!(
            extract_atom_link_href(xml),
            Some("https://sub.example.com/feed".to_string())
        );
    }

    #[test]
    fn test_decode_xml_entities() {
        assert_eq!(
            decode_xml_entities("Tom &amp; Jerry&#39;s &lt;show&gt;"),
            "Tom & Jerry's <show>"
        );
        // CDATA wrapper
        assert_eq!(
            decode_xml_entities("<![CDATA[The Real Title]]>"),
            "The Real Title"
        );
        // CDATA with entities inside — entities still get decoded (harmless in practice,
        // CDATA content rarely contains literal "&amp;")
        assert_eq!(
            decode_xml_entities("  <![CDATA[Spaced &amp; Raw]]>  "),
            "Spaced & Raw"
        );
        // Not CDATA — normal entity decode
        assert_eq!(decode_xml_entities("plain text"), "plain text");
    }

    #[test]
    fn test_strip_scripts() {
        let html = "<p>Hello</p><script>var x=1;</script><p>World</p><style>.a{}</style><p>!</p>";
        let cleaned = strip_scripts(html);
        assert_eq!(cleaned, "<p>Hello</p><p>World</p><p>!</p>");
    }

    #[test]
    fn test_extract_text_fallback() {
        let html = "<h1>Hello</h1><p>World <b>bold</b></p>";
        let text = extract_text_fallback(html);
        assert_eq!(text, "Hello World bold");
    }
}
