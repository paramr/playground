use tonic::{transport::Server, Request, Response, Status};

use proto::hello::hello_service_server::{HelloService, HelloServiceServer};
use proto::hello::{HelloResponse, HelloRequest};

#[derive(Debug, Default)]
pub struct HelloServiceImpl {}

#[tonic::async_trait]
impl HelloService for HelloServiceImpl {
    async fn hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = HelloResponse {
            message: format!("Hello {:?}!", request.into_inner().name).into(),
            payload: Some(shared::create_silly_payload()),
        };

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;
    let service = HelloServiceImpl::default();
    let service = HelloServiceServer::new(service);
    let service = tonic_web::config()
        .enable(service);
    Server::builder()
        .accept_http1(true)
        .add_service(service)
        .serve(addr)
        .await?;

    Ok(())
}
