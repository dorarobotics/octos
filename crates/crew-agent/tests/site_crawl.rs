//! Integration tests for DeepCrawlTool.
//!
//! These tests require a Chromium-compatible browser and network access.
//! Run with: cargo test -p crew-agent --test site_crawl -- --nocapture

use crew_agent::DeepCrawlTool;
use crew_agent::Tool;
use serde_json::json;

#[tokio::test]
async fn test_crawl_example_com() {
    let dir = tempfile::TempDir::new().unwrap();
    let tool = DeepCrawlTool::new(dir.path());

    let result = tool
        .execute(&json!({
            "url": "https://example.com",
            "max_depth": 1,
            "max_pages": 5
        }))
        .await
        .unwrap();

    assert!(result.success, "crawl failed: {}", result.output);
    assert!(result.output.contains("Deep Crawl:"));
    assert!(result.output.contains("Sitemap"));
    assert!(result.output.contains("example.com"));
    // Should have saved at least one page
    assert!(result.output.contains("pages saved to:"));
}

#[tokio::test]
async fn test_crawl_respects_max_pages() {
    let dir = tempfile::TempDir::new().unwrap();
    let tool = DeepCrawlTool::new(dir.path());

    let result = tool
        .execute(&json!({
            "url": "https://example.com",
            "max_depth": 2,
            "max_pages": 1
        }))
        .await
        .unwrap();

    assert!(result.success, "crawl failed: {}", result.output);
    // Output should say "Crawled 1 pages"
    assert!(
        result.output.contains("Crawled 1 pages"),
        "unexpected output: {}",
        result.output
    );
}

#[tokio::test]
async fn test_crawl_path_prefix() {
    let dir = tempfile::TempDir::new().unwrap();
    let tool = DeepCrawlTool::new(dir.path());

    let result = tool
        .execute(&json!({
            "url": "https://example.com",
            "max_depth": 2,
            "max_pages": 10,
            "path_prefix": "/nonexistent/"
        }))
        .await
        .unwrap();

    assert!(result.success);
    // With a nonexistent path prefix, only the seed page should be crawled
    assert!(result.output.contains("Crawled 1 pages"));
}
