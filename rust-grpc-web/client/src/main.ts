import App from './App.svelte'
import { WasmInit } from '../proto_build/wasm_api.client';

await WasmInit();

const app = new App({
  target: document.getElementById('app')
})

export default app
