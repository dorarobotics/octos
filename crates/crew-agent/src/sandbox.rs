//! Sandboxing for shell command execution.
//!
//! Provides platform-specific isolation: bubblewrap on Linux,
//! sandbox-exec on macOS, or no sandbox (pass-through).

use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// Sandbox configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Whether sandboxing is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Sandbox mode (auto-detect by default).
    #[serde(default)]
    pub mode: SandboxMode,

    /// Allow network access inside the sandbox.
    #[serde(default)]
    pub allow_network: bool,

    /// Docker-specific settings (used when mode = "docker").
    #[serde(default)]
    pub docker: DockerConfig,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: SandboxMode::Auto,
            allow_network: false,
            docker: DockerConfig::default(),
        }
    }
}

/// Docker sandbox configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DockerConfig {
    /// Docker image to use (default: "alpine:3.21").
    #[serde(default = "default_docker_image")]
    pub image: String,

    /// CPU limit (e.g. "1.0").
    #[serde(default)]
    pub cpu_limit: Option<String>,

    /// Memory limit (e.g. "512m").
    #[serde(default)]
    pub memory_limit: Option<String>,

    /// Maximum number of processes.
    #[serde(default)]
    pub pids_limit: Option<u32>,

    /// Workspace mount mode.
    #[serde(default)]
    pub mount_mode: MountMode,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            image: default_docker_image(),
            cpu_limit: None,
            memory_limit: None,
            pids_limit: None,
            mount_mode: MountMode::ReadWrite,
        }
    }
}

fn default_docker_image() -> String {
    "alpine:3.21".to_string()
}

/// Workspace mount mode for Docker sandbox.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MountMode {
    /// No workspace mount.
    None,
    /// Read-only mount.
    #[serde(rename = "ro")]
    ReadOnly,
    /// Read-write mount (default).
    #[default]
    #[serde(rename = "rw")]
    ReadWrite,
}

/// Which sandbox backend to use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxMode {
    /// Auto-detect: bwrap on Linux, sandbox-exec on macOS, none elsewhere.
    #[default]
    Auto,
    /// Linux bubblewrap.
    Bwrap,
    /// macOS sandbox-exec.
    Macos,
    /// Docker container isolation.
    Docker,
    /// No sandboxing (pass-through).
    None,
}

/// Trait for wrapping shell commands in a sandbox.
pub trait Sandbox: Send + Sync {
    /// Wrap a shell command string into a sandboxed `Command`.
    fn wrap_command(&self, shell_command: &str, cwd: &Path) -> Command;
}

/// No-op sandbox: executes commands directly.
pub struct NoSandbox;

impl Sandbox for NoSandbox {
    fn wrap_command(&self, shell_command: &str, cwd: &Path) -> Command {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(shell_command).current_dir(cwd);
        cmd
    }
}

/// Linux sandbox using bubblewrap (bwrap).
pub struct BwrapSandbox {
    allow_network: bool,
}

impl Sandbox for BwrapSandbox {
    fn wrap_command(&self, shell_command: &str, cwd: &Path) -> Command {
        let mut cmd = Command::new("bwrap");

        // Read-only bind system directories
        for dir in &["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"] {
            if Path::new(dir).exists() {
                cmd.arg("--ro-bind").arg(dir).arg(dir);
            }
        }

        // Read-write bind the working directory
        let cwd_str = cwd.to_string_lossy();
        cmd.arg("--bind").arg(&*cwd_str).arg(&*cwd_str);

        // Bind /tmp for scratch space
        cmd.arg("--tmpfs").arg("/tmp");

        // /dev minimal
        cmd.arg("--dev").arg("/dev");
        cmd.arg("--proc").arg("/proc");

        if !self.allow_network {
            cmd.arg("--unshare-net");
        }

        cmd.arg("--unshare-pid");
        cmd.arg("--die-with-parent");
        cmd.arg("--chdir").arg(&*cwd_str);
        cmd.arg("--").arg("sh").arg("-c").arg(shell_command);

        cmd
    }
}

/// macOS sandbox using sandbox-exec.
pub struct MacosSandbox {
    allow_network: bool,
}

impl Sandbox for MacosSandbox {
    fn wrap_command(&self, shell_command: &str, cwd: &Path) -> Command {
        let cwd_str = cwd.to_string_lossy();

        // Reject paths with control characters to prevent SBPL profile injection.
        // Fail closed: return a command that exits with error instead of running unsandboxed.
        if cwd_str.bytes().any(|b| b < 0x20 || b == 0x7F) {
            tracing::error!("cwd contains control characters, refusing to execute");
            let mut cmd = Command::new("sh");
            cmd.arg("-c").arg("echo 'sandbox error: cwd contains invalid characters' >&2; exit 1");
            return cmd;
        }

        // Escape characters that could break the SBPL sandbox profile syntax
        let cwd_escaped = cwd_str.replace('\\', "\\\\").replace('"', "\\\"");

        let network_rule = if self.allow_network {
            "(allow network*)"
        } else {
            "(deny network*)"
        };

        let profile = format!(
            r#"(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow file-read*)
(allow file-write* (subpath "{cwd}"))
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (subpath "/private/var/folders"))
{network_rule}
"#,
            cwd = cwd_escaped,
            network_rule = network_rule,
        );

        let mut cmd = Command::new("sandbox-exec");
        cmd.arg("-p")
            .arg(profile)
            .arg("sh")
            .arg("-c")
            .arg(shell_command)
            .current_dir(cwd);
        cmd
    }
}

/// Docker container sandbox.
pub struct DockerSandbox {
    config: DockerConfig,
    allow_network: bool,
}

impl Sandbox for DockerSandbox {
    fn wrap_command(&self, shell_command: &str, cwd: &Path) -> Command {
        let mut cmd = Command::new("docker");
        cmd.arg("run").arg("--rm");

        // Resource limits
        if let Some(ref cpu) = self.config.cpu_limit {
            cmd.arg("--cpus").arg(cpu);
        }
        if let Some(ref mem) = self.config.memory_limit {
            cmd.arg("--memory").arg(mem);
        }
        if let Some(pids) = self.config.pids_limit {
            cmd.arg("--pids-limit").arg(pids.to_string());
        }

        // Network
        if !self.allow_network {
            cmd.arg("--network").arg("none");
        }

        // Security hardening
        cmd.arg("--security-opt").arg("no-new-privileges");
        cmd.arg("--cap-drop").arg("ALL");

        // Clear dangerous environment variables
        for var in &["LD_PRELOAD", "LD_LIBRARY_PATH", "NODE_OPTIONS", "PYTHONSTARTUP"] {
            cmd.arg("--env").arg(format!("{var}="));
        }

        // Workspace mount — validate path to prevent volume mount injection via ':'
        let cwd_str = cwd.to_string_lossy();
        if cwd_str.contains(':') || cwd_str.contains('\0') {
            tracing::error!("cwd contains invalid characters for Docker mount, refusing to execute");
            let mut fail = Command::new("sh");
            fail.arg("-c").arg("echo 'sandbox error: cwd contains invalid characters' >&2; exit 1");
            return fail;
        }
        match self.config.mount_mode {
            MountMode::ReadWrite => {
                cmd.arg("-v").arg(format!("{cwd_str}:/workspace"));
                cmd.arg("-w").arg("/workspace");
            }
            MountMode::ReadOnly => {
                cmd.arg("-v").arg(format!("{cwd_str}:/workspace:ro"));
                cmd.arg("-w").arg("/workspace");
            }
            MountMode::None => {
                cmd.arg("-w").arg("/tmp");
            }
        }

        cmd.arg(&self.config.image);
        cmd.arg("sh").arg("-c").arg(shell_command);
        cmd
    }
}

/// Create a sandbox from config.
pub fn create_sandbox(config: &SandboxConfig) -> Box<dyn Sandbox> {
    if !config.enabled {
        tracing::info!("sandbox disabled, shell commands run without isolation");
        return Box::new(NoSandbox);
    }

    match &config.mode {
        SandboxMode::None => Box::new(NoSandbox),
        SandboxMode::Bwrap => Box::new(BwrapSandbox {
            allow_network: config.allow_network,
        }),
        SandboxMode::Macos => Box::new(MacosSandbox {
            allow_network: config.allow_network,
        }),
        SandboxMode::Docker => Box::new(DockerSandbox {
            config: config.docker.clone(),
            allow_network: config.allow_network,
        }),
        SandboxMode::Auto => {
            if cfg!(target_os = "linux") && which_exists("bwrap") {
                Box::new(BwrapSandbox {
                    allow_network: config.allow_network,
                })
            } else if cfg!(target_os = "macos") && which_exists("sandbox-exec") {
                Box::new(MacosSandbox {
                    allow_network: config.allow_network,
                })
            } else if which_exists("docker") {
                Box::new(DockerSandbox {
                    config: config.docker.clone(),
                    allow_network: config.allow_network,
                })
            } else {
                Box::new(NoSandbox)
            }
        }
    }
}

/// Check if a binary exists on PATH.
fn which_exists(bin: &str) -> bool {
    std::process::Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_sandbox_wraps_directly() {
        let sb = NoSandbox;
        let cmd = sb.wrap_command("echo hello", Path::new("/tmp"));
        // Command should be `sh -c "echo hello"` in /tmp
        let prog = cmd.as_std().get_program().to_string_lossy().to_string();
        assert_eq!(prog, "sh");
    }

    #[test]
    fn test_create_sandbox_disabled() {
        let config = SandboxConfig::default();
        let sb = create_sandbox(&config);
        // Should be NoSandbox — just verify it doesn't panic
        let _cmd = sb.wrap_command("ls", Path::new("/tmp"));
    }

    #[test]
    fn test_bwrap_sandbox_command() {
        let sb = BwrapSandbox {
            allow_network: false,
        };
        let cmd = sb.wrap_command("echo hi", Path::new("/tmp/test"));
        let prog = cmd.as_std().get_program().to_string_lossy().to_string();
        assert_eq!(prog, "bwrap");
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(args.contains(&"echo hi".to_string()));
    }

    #[test]
    fn test_macos_sandbox_command() {
        let sb = MacosSandbox {
            allow_network: true,
        };
        let cmd = sb.wrap_command("echo hi", Path::new("/tmp/test"));
        let prog = cmd.as_std().get_program().to_string_lossy().to_string();
        assert_eq!(prog, "sandbox-exec");
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.iter().any(|a| a.contains("allow network")));
    }

    #[test]
    fn test_docker_sandbox_command() {
        let sb = DockerSandbox {
            config: DockerConfig::default(),
            allow_network: false,
        };
        let cmd = sb.wrap_command("echo hi", Path::new("/tmp/work"));
        let prog = cmd.as_std().get_program().to_string_lossy().to_string();
        assert_eq!(prog, "docker");
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"run".to_string()));
        assert!(args.contains(&"--rm".to_string()));
        assert!(args.contains(&"none".to_string())); // --network none
        assert!(args.contains(&"no-new-privileges".to_string()));
        assert!(args.contains(&"ALL".to_string())); // --cap-drop ALL
        assert!(args.contains(&"alpine:3.21".to_string()));
        assert!(args.contains(&"echo hi".to_string()));
    }

    #[test]
    fn test_docker_sandbox_resource_limits() {
        let sb = DockerSandbox {
            config: DockerConfig {
                cpu_limit: Some("1.5".into()),
                memory_limit: Some("256m".into()),
                pids_limit: Some(100),
                ..DockerConfig::default()
            },
            allow_network: true,
        };
        let cmd = sb.wrap_command("ls", Path::new("/tmp"));
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"1.5".to_string())); // --cpus
        assert!(args.contains(&"256m".to_string())); // --memory
        assert!(args.contains(&"100".to_string())); // --pids-limit
        // Network allowed — no --network none
        assert!(!args.iter().any(|a| a == "none"));
    }

    #[test]
    fn test_docker_sandbox_mount_modes() {
        // Read-write (default)
        let sb = DockerSandbox {
            config: DockerConfig::default(),
            allow_network: false,
        };
        let cmd = sb.wrap_command("ls", Path::new("/home/user/project"));
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"/home/user/project:/workspace".to_string()));

        // Read-only
        let sb_ro = DockerSandbox {
            config: DockerConfig {
                mount_mode: MountMode::ReadOnly,
                ..DockerConfig::default()
            },
            allow_network: false,
        };
        let cmd_ro = sb_ro.wrap_command("ls", Path::new("/home/user/project"));
        let args_ro: Vec<_> = cmd_ro
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args_ro.contains(&"/home/user/project:/workspace:ro".to_string()));

        // No mount — uses /tmp as workdir instead of /workspace
        let sb_none = DockerSandbox {
            config: DockerConfig {
                mount_mode: MountMode::None,
                ..DockerConfig::default()
            },
            allow_network: false,
        };
        let cmd_none = sb_none.wrap_command("ls", Path::new("/home/user/project"));
        let args_none: Vec<_> = cmd_none
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(!args_none.iter().any(|a| a.contains(":/workspace")));
        assert!(args_none.contains(&"/tmp".to_string())); // -w /tmp
    }

    #[test]
    fn test_docker_sandbox_rejects_colon_in_path() {
        let sb = DockerSandbox {
            config: DockerConfig::default(),
            allow_network: false,
        };
        let cmd = sb.wrap_command("ls", Path::new("/tmp/evil:/host"));
        let prog = cmd.as_std().get_program().to_string_lossy().to_string();
        assert_eq!(prog, "sh"); // falls back to error command
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.iter().any(|a| a.contains("exit 1")));
    }

    #[test]
    fn test_macos_sandbox_rejects_control_chars() {
        let sb = MacosSandbox {
            allow_network: false,
        };
        let cmd = sb.wrap_command("ls", Path::new("/tmp/\x01bad"));
        let prog = cmd.as_std().get_program().to_string_lossy().to_string();
        assert_eq!(prog, "sh"); // error command, not sandbox-exec
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        // Must NOT execute the original command unsandboxed
        assert!(args.iter().any(|a| a.contains("exit 1")));
        assert!(!args.iter().any(|a| a.contains("ls")));
    }

    #[test]
    fn test_docker_sandbox_env_sanitization() {
        let sb = DockerSandbox {
            config: DockerConfig::default(),
            allow_network: false,
        };
        let cmd = sb.wrap_command("ls", Path::new("/tmp"));
        let args: Vec<_> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"LD_PRELOAD=".to_string()));
        assert!(args.contains(&"LD_LIBRARY_PATH=".to_string()));
        assert!(args.contains(&"NODE_OPTIONS=".to_string()));
        assert!(args.contains(&"PYTHONSTARTUP=".to_string()));
    }
}
