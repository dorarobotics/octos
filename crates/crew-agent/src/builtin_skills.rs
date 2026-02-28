//! Built-in skills bundled with crew-rs at compile time.

/// A built-in skill with name and full SKILL.md content (including frontmatter).
pub struct BuiltinSkill {
    pub name: &'static str,
    pub content: &'static str,
}

/// All built-in skills, embedded at compile time.
pub const BUILTIN_SKILLS: &[BuiltinSkill] = &[
    BuiltinSkill {
        name: "cron",
        content: include_str!("../skills/cron/SKILL.md"),
    },
    BuiltinSkill {
        name: "github",
        content: include_str!("../skills/github/SKILL.md"),
    },
    BuiltinSkill {
        name: "news",
        content: include_str!("../skills/news/SKILL.md"),
    },
    BuiltinSkill {
        name: "skill-creator",
        content: include_str!("../skills/skill-creator/SKILL.md"),
    },
    BuiltinSkill {
        name: "summarize",
        content: include_str!("../skills/summarize/SKILL.md"),
    },
    BuiltinSkill {
        name: "tmux",
        content: include_str!("../skills/tmux/SKILL.md"),
    },
    BuiltinSkill {
        name: "weather",
        content: include_str!("../skills/weather/SKILL.md"),
    },
];
