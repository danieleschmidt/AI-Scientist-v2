# Rerere Cache Audit

This directory contains audit information for Git's rerere (reuse recorded resolution) cache.

## Purpose

Git rerere automatically records and reapplies conflict resolutions. This audit mechanism helps track:
- What conflicts have been automatically resolved
- When resolution patterns were recorded
- Verification that auto-resolutions are correct

## Usage

Run `git rerere diff` to see current resolution patterns.
CI will automatically upload rerere diffs as artifacts for review.

## Files

- `rerere-audit.log` - Log of all automatic conflict resolutions
- `*.diff` - Specific resolution patterns for review