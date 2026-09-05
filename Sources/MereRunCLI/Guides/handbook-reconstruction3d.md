# TripoSR, InstantMesh, and TRELLIS.2

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare reference views for TripoSR, InstantMesh, and TRELLIS.2.

## Start here

Choose the reconstruction engine, then provide the source views it expects. Use
clear object silhouettes and a consistent subject. Input images carry the shape
evidence; prose can't repair unseen or contradictory surfaces.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Reference example:
A centered object with its full outline visible and a simple background. For
multiview reconstruction, provide different ordered views of the
same unchanged object.
```

## Controls and variants

Use each engine's local workflow for view count, ordering, background handling,
and export controls. Preserve generated geometry and textures together. A
single-view reconstruction invents unseen surfaces, so review the rear and
underside.

## Iterate and review

If geometry has holes or duplicate parts, inspect occlusion and inconsistent
views. If textures look good from one angle only, orbit the result. Validate
scale, topology, and material files before using the asset in another tool.

## Read this guide offline

Reading the handbook requires neither model weights nor a network connection.
Before running inference offline, download the checkpoint and any optional
components that your workflow uses.

To read the guide in macOS Studio, follow these steps:

1. In **Help**, select **mere.run Guide**.
2. In **Guide collection**, select **Models**.
3. In the search field, enter a family name or model ID.
4. Select the guide.
5. In **Model**, select a variant.

Prompt examples include a **Copy** control. Input examples serve as
file-preparation checklists. Reading a guide doesn't start inference.

To read this guide in a terminal, run the following command:

```bash
mere.run guide --model image-3d-triposr
```

To inspect the available command options, run the following command:

```bash
mere.run vision image-to-3d --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-3d-triposr`
- `image-3d-instantmesh-base`
- `image-3d-trellis2-4b`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Vision runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
