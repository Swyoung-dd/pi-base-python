# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

piY is for developers who want to run a local AI coding agent from a browser while staying close to their project files and existing terminal sessions.

## Product Purpose

piY Web provides a complete browser workspace for the Python-based piY coding agent. Success means a developer can resume sessions, send prompts with live progress, change models, browse project files, and inspect results without returning to the terminal for routine work.

## Positioning

piY Web is the native browser surface for piY's Python runtime and on-disk `.piy` data. It exposes the agent's existing session, model, tool, and file workflows instead of introducing a separate service or incompatible conversation format.

## Operating Context

Developers run piY locally from a project directory. Sessions and configuration live under `.piy`; the Web interface operates against the selected local project and is intended for repeated, work-focused use on desktop, with a usable mobile layout for review and lightweight prompting.

## Capabilities and Constraints

- The first complete release includes session browsing, live chat, model selection, project file browsing and preview, and light/dark themes.
- Existing piY session and configuration formats remain the source of truth.
- The interface runs locally and must not expose arbitrary filesystem paths outside the selected project.
- The existing terminal interface remains available and is not replaced.
- Web-specific implementation belongs in this repository and must integrate with the Python agent runtime.

## Brand Commitments

- The product name is `piY`.
- The interface is inspired functionally by the neighboring `pi-web` project but must have its own identity rather than copy its appearance.
- Product copy should be concise, technical, and suitable for developers.

## Evidence on Hand

- Existing runtime, model, tool, session, and configuration implementations under `src/pi`.
- Existing local session data under `.piy/sessions` for development validation.
- `D:\agent\pi-web` is a functional reference for expected workflows and feature coverage, not a visual template.
- No customer claims, benchmarks, pricing, testimonials, or production deployment evidence are available and none should be fabricated.

## Product Principles

- Keep the project and the conversation visible together.
- Make agent state legible while work is running.
- Preserve local data ownership and existing piY formats.
- Prefer direct, compact controls over explanatory interface copy.
- Keep terminal and Web workflows interoperable.

## Accessibility & Inclusion

Keyboard access, visible focus, reduced-motion support, readable contrast, and responsive layouts are required for the primary workflows.
