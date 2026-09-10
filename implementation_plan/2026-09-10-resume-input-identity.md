# Resume input identity and manual-session recovery

## Problem and goal
Multi-day reuse rejects metadata drift in unused per-recording amplifier XML files even when an explicit authoritative XML is selected. GUI recovery interprets a Phy manual lease as a persistent Run and fails.

## Changes
- Record content hashes for small configuration sidecars; accept timestamp drift only with matching recorded and live content hashes. Keep binary acquisition metadata checks and legacy checks without hashes.
- Exclude per-recording amplifier.xml from multi-day compatibility when an explicit authoritative XML is configured, matching multiday staging semantics. Continue tracking settings.xml and the authoritative XML.
- Recover GUI settings through a manual lease's previous Run claim or completed record without removing the lease.

## Verification
Regression tests for metadata-only XML changes, changed XML contents, binary changes, unused multi-day XML, and manual-lease GUI recovery. Read-only compatibility check against the actual Day10–Day245 snapshots. No preprocessing or sorting jobs launched.
