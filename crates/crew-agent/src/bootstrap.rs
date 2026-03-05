//! Bootstrap bundled app-skill binaries into the skills directory.
//!
//! At gateway startup, copies sibling binaries (built alongside `crew`) into
//! `.crew/skills/<dir>/main`, plus writes the embedded SKILL.md and manifest.json.

use std::path::Path;

use crate::bundled_app_skills::{BUNDLED_APP_SKILLS, PLATFORM_SKILLS};

/// Bootstrap all bundled app-skills into `skills_dir`.
///
/// For each skill, if `skills_dir/<dir>/main` does not already exist, creates the
/// directory and writes SKILL.md, manifest.json, and copies the sibling binary.
///
/// Returns the number of skills bootstrapped.
pub fn bootstrap_bundled_skills(skills_dir: &Path) -> usize {
    let exe_dir = match std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        Some(d) => d,
        None => return 0,
    };

    let mut count = 0;

    for &(dir_name, binary_name, skill_md, manifest_json) in BUNDLED_APP_SKILLS.iter().chain(PLATFORM_SKILLS.iter()) {
        let skill_dir = skills_dir.join(dir_name);
        let main_path = skill_dir.join("main");

        // Skip if already bootstrapped
        if main_path.exists() {
            continue;
        }

        // Find sibling binary
        let src_binary = exe_dir.join(binary_name);
        if !src_binary.exists() {
            continue;
        }

        // Create skill directory
        if std::fs::create_dir_all(&skill_dir).is_err() {
            continue;
        }

        // Write SKILL.md
        if std::fs::write(skill_dir.join("SKILL.md"), skill_md).is_err() {
            continue;
        }

        // Write manifest.json
        if std::fs::write(skill_dir.join("manifest.json"), manifest_json).is_err() {
            continue;
        }

        // Copy binary as "main"
        if std::fs::copy(&src_binary, &main_path).is_err() {
            continue;
        }

        // chmod 755 on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&main_path, std::fs::Permissions::from_mode(0o755));
        }

        count += 1;
    }

    count
}

/// Bootstrap a single named app-skill into `skills_dir`.
///
/// Unlike `bootstrap_bundled_skills`, this always overwrites existing files
/// (used for conditional skills that may need re-bootstrap after updates).
///
/// Returns `true` if the skill was successfully bootstrapped.
pub fn bootstrap_single_skill(skills_dir: &Path, name: &str) -> bool {
    let exe_dir = match std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        Some(d) => d,
        None => return false,
    };

    let entry = BUNDLED_APP_SKILLS
        .iter()
        .chain(PLATFORM_SKILLS.iter())
        .find(|&&(dir_name, _, _, _)| dir_name == name);

    let &(dir_name, binary_name, skill_md, manifest_json) = match entry {
        Some(e) => e,
        None => return false,
    };

    let skill_dir = skills_dir.join(dir_name);
    let main_path = skill_dir.join("main");

    let src_binary = exe_dir.join(binary_name);
    if !src_binary.exists() {
        return false;
    }

    if std::fs::create_dir_all(&skill_dir).is_err() {
        return false;
    }

    if std::fs::write(skill_dir.join("SKILL.md"), skill_md).is_err() {
        return false;
    }
    if std::fs::write(skill_dir.join("manifest.json"), manifest_json).is_err() {
        return false;
    }
    if std::fs::copy(&src_binary, &main_path).is_err() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&main_path, std::fs::Permissions::from_mode(0o755));
    }

    true
}
