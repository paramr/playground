#! /bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_PATH=${SCRIPT_PATH}
SRC_PROTO_PATH="${ROOT_PATH}/../../proto"

PROTO_GEN_DIR="${ROOT_PATH}/../proto_build"

PROTO_PATHS=(
  "data"
)

INC_OPT=
for path in "${PROTO_PATHS[@]}";
do
  INC_OPT="${INC_OPT} -I${SRC_PROTO_PATH}/${path}"
done
INC_OPT="${INC_OPT} -I${SCRIPT_PATH}"

echo "Wasm pack build..."
wasm-pack build ./wasm --target web

# https://github.com/timostamm/protobuf-ts/blob/master/MANUAL.md
echo "Generating ts proto structs..."
npx protoc \
  ${INC_OPT} \
  --ts_out "${PROTO_GEN_DIR}" \
  --ts_opt force_server_none,force_client_none,optimize_code_size \
  wasm_api.proto

echo "Generating ts shim..."
protoc \
  ${INC_OPT} \
  --plugin=protoc-gen-wasm_ts_wrapper-plugin="${SCRIPT_PATH}/wasm_ts_wrapper_plugin.py" \
  --wasm_ts_wrapper-plugin_out="${PROTO_GEN_DIR}" \
  wasm_api.proto
