<!-- project-control:start -->
## Project Control and Todo workflow

Use Project Control's existing `overview`, `search`, `delta`, `frontier`, `history`,
`evidence`, and `impact` reads for registered-workspace context and migrated Todo
status/history. The compact profile catalogs are observer 11, coder 12, and
mutator 16 tools. Routine workflow begins with
`next_task(project, repository?, task_id?, run_id?)`; pass its returned handle to
`inspect_task`, `coordinate_task`, and `finish_task`. Use the live host manifest
for exact schemas and profile permissions.

Project Control binds workflow handles to the registered workspace, identity,
profile, credentials, and providers. Routine agents should not manage native
Todo IDs, leases, credentials, or retry keys, and do not pass raw repository
paths as selectors. Privileged plan changes, execution maintenance, and project
metadata stay on their established routes. Observer reads need no task claim and
cannot mutate. The standalone Todo API and CLI remain available for separately
managed authorities. Imported completion is a source assertion, not native
verification; unknown legacy effects require attention and are never replayed
automatically. Existing non-Todo tools remain available for their own purposes.
<!-- project-control:end -->
