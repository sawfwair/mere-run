<script setup lang="ts">
import { withBase } from 'vitepress'
import proofMediaUrl from '../../../../media/demo-inline.gif'
</script>

<template>
  <section class="mere-launch-stage" aria-label="Real mere.run session on an M4 Max">
    <div class="mere-launch-frame">
      <div class="mere-launch-topline">
        <div class="mere-window-controls" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <span class="mere-launch-title">mere.run / local session</span>
        <span class="mere-launch-status">running</span>
      </div>

      <div class="mere-launch-media">
        <img
          :src="proofMediaUrl"
          width="900"
          height="630"
          alt="A real terminal session generating text, images, 3D, audio, and video locally with mere.run"
        >
        <span class="mere-proof-label">Real output · M4 Max</span>
      </div>

      <div class="mere-launch-command">
        <span aria-hidden="true">$</span>
        <code>mere.run model capabilities --recommended</code>
        <a :href="withBase('/runtime/image')">Explore runtimes <span aria-hidden="true">↗</span></a>
      </div>
    </div>
  </section>
</template>
