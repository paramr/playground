<script lang="ts">
  import {GrpcWebFetchTransport} from "@protobuf-ts/grpcweb-transport";
  import * as server_api_pb from '../proto_build/server_api';
  import * as server_api_client from '../proto_build/server_api.client';
  import * as wasm_api_pb from '../proto_build/wasm_api';
  import * as wasm_api_client from '../proto_build/wasm_api.client';

  let wasmReq: wasm_api_pb.WasmRequest = {
    name: 'Grpc web test',
  };
  let wasmResp = wasm_api_client.WasmServiceHello(wasmReq);
  let req: server_api_pb.HelloRequest = {
    name: 'Grpc web test',
  };
  const transport = new GrpcWebFetchTransport({
    baseUrl: 'http://localhost:50051',
  });
  const helloClient = new server_api_client.HelloServiceClient(transport);
</script>

<main>
	<h1>Hello!</h1>
  <p>The wasm response is {wasm_api_pb.WasmResponse.toJsonString(wasmResp)}</p>
  {#await helloClient.hello(req)}
  	<p>Waiting...</p>
  {:then resp}
    <p>The server response is {server_api_pb.HelloResponse.toJsonString(resp.response)}</p>
  {:catch error}
    <p style="color: red">{error.message}</p>
  {/await}
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>