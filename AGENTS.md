<!-- project-control:start -->
## Project Control and native Todo workflow

Use Project Control's `overview`, `search`, `delta`, `frontier`, `history`,
`evidence`, and `impact` tools for registered-workspace context. They are not
the Todo transaction API. For Todo state and workflow, use the native `todo_*`
tools exposed by the host. Start with `todo_read_authority`; for plan changes,
call `todo_preview_plan`, inspect the returned preview, then pass that exact
preview to `todo_commit_plan` with an idempotency key. Use the corresponding
`todo_open_run`, `todo_create_workspace`, `todo_dispatch_attempt`, outcome,
verification, and recovery tools for later lifecycle steps.

Arguments use a registered workspace ID and, when necessary, an optional
registered `repository` alias. The host binds authority, identity, credentials,
profile, and providers; never pass filesystem roots or bearer credentials as
selectors. Read migrated history with `todo_search_legacy_history`, current
status with `todo_read_legacy_current_tasks`, and preserved declarations with
`todo_read_legacy_project_context`. Imported completion is a source assertion,
not native verification; unknown legacy effects require attention and are never
replayed automatically. Project Control's existing non-Todo tools remain
available for their own purposes.
<!-- project-control:end -->
