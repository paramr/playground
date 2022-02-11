<script lang="ts">
  import {GrpcWebFetchTransport} from "@protobuf-ts/grpcweb-transport";
  import * as api_pb from '../proto/build/api';
  import * as api_client from '../proto/build/api.client';
  import { greet } from '../wasm/pkg/wasm';

  greet('from vite!');
  let req: api_pb.HelloRequest = {
    name: 'Grpc web test',
  };
  const transport = new GrpcWebFetchTransport({
    baseUrl: 'http://localhost:50051',
  });
  const helloClient = new api_client.HelloServiceClient(transport);
</script>

<main>
	<h1>Hello!</h1>
  {#await helloClient.hello(req)}
  	<p>Waiting...</p>
  {:then resp}
    <p>The response is {api_pb.HelloResponse.toJsonString(resp.response)}</p>
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