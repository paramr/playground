<script lang="ts">
  import * as hello_pb from '../proto/build/hello_pb';
  import * as hello_svc from '../proto/build/hello_grpc_web_pb';
	export let name: string;

  let req = new hello_pb.HelloRequest();
  req.setName('Grpc web test');
  const helloService = new hello_svc.HelloServicePromiseClient('http://localhost:50051');
</script>

<main>
	<h1>Hello {name}!</h1>
  {#await helloService.hello(req)}
  	<p>Waiting...</p>
  {:then resp}
    <p>The number is {resp.getMessage()}</p>
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