#! /bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_PATH=${SCRIPT_PATH}
SRC_PROTO_PATH="${ROOT_PATH}/../proto"

PROTO_GEN_DIR="${ROOT_PATH}/proto_build"

rm -Rf ${PROTO_GEN_DIR}
mkdir -p ${PROTO_GEN_DIR}

PROTO_PATHS=(
  "data"
  "server_api"
)

INC_OPT=
for path in "${PROTO_PATHS[@]}";
do
  INC_OPT="${INC_OPT} -I${SRC_PROTO_PATH}/${path}"
done

PROTO_FILES=(
  "data"
  "server_api"
)

# https://github.com/timostamm/protobuf-ts/blob/master/MANUAL.md

for file in "${PROTO_FILES[@]}";
do
echo "Building: ${file}"
npx protoc \
  ${INC_OPT} \
  --ts_out ${PROTO_GEN_DIR} \
  --ts_opt client_generic,optimize_code_size \
  "${file}.proto"
done
