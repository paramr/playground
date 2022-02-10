#! /bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_PATH=${SCRIPT_PATH}
SRC_PROTO_PATH="${ROOT_PATH}/../proto"

# Directory to write generated code to (.js and .d.ts files)
PROTO_GEN_DIR="${ROOT_PATH}/proto/build"

mkdir -p ${PROTO_GEN_DIR}

PROTO_FILES=(
  "hello"
  "shared"
)

for file in "${PROTO_FILES[@]}";
do
echo "Building: ${file}"
npx protoc \
  -I "${SRC_PROTO_PATH}" \
  --ts_out ${PROTO_GEN_DIR} \
  --ts_opt client_generic,optimize_code_size \
    "${file}.proto"
done
