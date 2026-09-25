# Outcome-blind development image review

Review the acquired images in the hash order frozen before acquisition. The target is the first 256 eligible distinct scenes in that order, after the exclusions below. Reserved test annotations remain untouched. Do not inspect model outputs, choose easy questions, rebalance based on observed accuracy, or change labels to match predictions.

Exclude only a broken/placeholder image, a clear mismatch with its situation annotation, or a duplicate underlying photograph/scene already represented in the old panel or an earlier retained development candidate. Record the image ID, reason, source digest and review evidence. The verified old-panel duplicate viva-831 is already excluded. Similar generic subject matter alone is not proof of a shared scene.

Keep uncertain or interpretive situations with an explicit flag unless a listed exclusion is established. Cartoon/staged/stock images, watermarks and subjective official labels are dataset properties, not automatic exclusion reasons. An unresolvable small thumbnail requires full-image review before a decision. Labels remain the released labels.

Contact sheets are review aids, not model input. They show ID, frozen rank, image and situation description. The review is by the coding assistant, not an independent human annotation panel, and must be reported as such. Pixel checks and dHash screening cannot prove absence of shared source scenes; residual uncertainty must remain in the paper.

Freeze the final ID list, image hashes, prompt, parser, shuffled-image mapping and endpoint model identities only after review. Do not run endpoint outcomes until that manifest and the execution budget are saved. If fewer than 256 pass, report the shortfall before revising the panel size or candidate pool.
