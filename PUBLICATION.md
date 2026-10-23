# Publication notes

Video Stylizer for Apple Silicon is being preserved as a public-source candidate.

Current scope: Python video pipeline plus native Swift/Core ML backend, setup scripts, tests and verification notes.

Preparation still required: Exclude coreml/.build, .venv, runs and input footage; input/ is not currently excluded by the root ignore file. Replace personal model paths with configuration and document how users obtain their own model. Add a license. Verification notes include a disk-space interruption, so do a fresh clean-machine smoke check.

This source export deliberately omits local credentials, personal configurations, caches, installed dependencies, local datasets and selected media/models. Some demos need replacement assets before they run. Existing validation documents describe earlier checks and have not been rerun for this export. No new open-source license is granted by this export; see any existing project license.
