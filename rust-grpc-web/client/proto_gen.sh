#! /bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_PATH=${SCRIPT_PATH}
SRC_PROTO_PATH="${ROOT_PATH}/../proto"

# PROTOC_GEN_TS_PROTO_PATH="${ROOT_PATH}/node_modules/.bin/protoc-gen-ts_proto"

# Directory to write generated code to (.js and .d.ts files)
PROTO_GEN_DIR="${ROOT_PATH}/proto/build"

# echo "$SCRIPT_PATH"
# echo "$ROOT_PATH"
# echo "$SRC_PROTO_PATH"
# echo "$PROTOC_GEN_TS_PROTO_PATH"
# echo "$PROTO_GEN_DIR"

mkdir -p ${PROTO_GEN_DIR}

PROTO_FILES=(
  "hello"
)

for file in "${PROTO_FILES[@]}";
do
echo "Building: ${file}"
protoc \
    --proto_path="${SRC_PROTO_PATH}" \
    --js_out=import_style=commonjs:${PROTO_GEN_DIR} \
    --grpc-web_out=import_style=commonjs+dts,mode=grpcwebtext:${PROTO_GEN_DIR} \
    "${file}.proto"
done

