## Purpose

Provide consistent, readable Mapal source formatting through the command line, Neovim and VS Code while preserving program meaning and comments.

## ADDED Requirements

### Requirement: Stable source layout
The formatter SHALL use four-space indentation, multiline brace bodies, separate statements and record fields onto lines, consistent token spacing, and a final newline for nonempty source. It MUST preserve tokens, literal spelling and comments, and formatting its own output MUST leave it unchanged.

#### Scenario: Nested compact program
- **WHEN** a valid program contains inline and nested brace bodies with inconsistent indentation
- **THEN** formatting expands those bodies and produces stable indentation while preserving tokens and comments

#### Scenario: Malformed input
- **WHEN** input contains a syntax error or unsupported lexical construct
- **THEN** formatting fails with diagnostics and supplies no partial replacement

### Requirement: CLI formatting modes
The CLI SHALL provide `mapal fmt <paths...>` for in-place formatting, `--check` for non-writing verification, `--stdout` for a single-file preview, and `-` for stdin/stdout. It MUST validate every requested input before writing any file and MUST NOT require a compiler toolchain to format source.

#### Scenario: Format and check files
- **WHEN** explicit valid files are formatted and then checked
- **THEN** formatting updates those files and checking returns success without writing

#### Scenario: Formatting drift
- **WHEN** a noncanonical file is checked
- **THEN** the command exits unsuccessfully, identifies the file, and leaves it unchanged

#### Scenario: Invalid or unreadable batch input
- **WHEN** one requested file is invalid or unreadable
- **THEN** the command fails and no requested file is formatted

#### Scenario: Buffer and preview modes
- **WHEN** source is passed on stdin or one file is previewed with `--stdout`
- **THEN** formatted source alone appears on stdout and no file is written

### Requirement: Editor formatting shares CLI behavior
Neovim and VS Code SHALL expose whole-document formatting through the CLI, accept a configured executable path, and format unsaved buffer contents. The plugins MUST apply no replacement on subprocess failure or to a stale buffer version.

#### Scenario: Format an unsaved document
- **WHEN** a user invokes the plugin's formatting action on valid unsaved Mapal text
- **THEN** the buffer receives the same output as stdin formatting through the CLI

#### Scenario: Missing CLI or invalid source
- **WHEN** the formatter executable is unavailable or source formatting fails
- **THEN** the editor reports the error and preserves the original buffer
