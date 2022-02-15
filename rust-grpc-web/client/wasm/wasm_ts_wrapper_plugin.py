#!/usr/bin/env python3

# Learnt from:
# https://manzan.dev/python-protoc-plugin-2

import sys
import re

from google.protobuf.compiler import plugin_pb2 as plugin

to_snake_pattern1 = re.compile(r'(.)([A-Z][a-z]+)')
to_snake_pattern2 = re.compile(r'([a-z0-9])([A-Z])')

def camel_to_snake(name):
  name = to_snake_pattern1.sub(r'\1_\2', name)
  name = to_snake_pattern2.sub(r'\1_\2', name).lower()
  return name

def process(
    request: plugin.CodeGeneratorRequest,
    response: plugin.CodeGeneratorResponse
) -> None:
  response.supported_features = plugin.CodeGeneratorResponse.FEATURE_PROTO3_OPTIONAL
  for proto_file in request.proto_file:
    has_service = False
    file = plugin.CodeGeneratorResponse.File()
    proto_file_name = proto_file.name.split('.')[0]
    file.name = "{}.client.ts".format(proto_file_name)
    file_content = ""
    file_content += f"import * as {proto_file_name} from './{proto_file_name}';\n"
    file_content += "import init from '../wasm/pkg/wasm';\n"
    file_content += "import * as wasm from '../wasm/pkg/wasm';\n\n"
    file_content += "export async function WasmInit() {\n"
    file_content += "  await init();\n"
    file_content += "}\n\n"
    for service in proto_file.service:
      has_service = True
      service_name = service.name
      for method in service.method:
        # type names pave . as prefix
        input_type = method.input_type[1:]
        output_type = method.output_type[1:]
        snake_method_name = camel_to_snake(method.name)
        file_content += f"export function {service_name}{method.name}(input: {input_type}): {output_type} {{\n"
        file_content += f"  let input_bytes = {input_type}.toBinary(input);\n"
        file_content += f"  let output_bytes = wasm.{snake_method_name}(input_bytes);\n"
        file_content += f"  return {output_type}.fromBinary(output_bytes);\n"
        file_content += "}\n\n"

    if has_service:
      file.content = file_content
      response.file.append(file)


def main() -> None:
  request = plugin.CodeGeneratorRequest.FromString(sys.stdin.buffer.read())
  response = plugin.CodeGeneratorResponse()
  process(request, response)
  sys.stdout.buffer.write(response.SerializeToString())

if __name__ == "__main__":
  main()
