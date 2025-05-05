#include <nan.h>
#include <node.h>
#include <node_buffer.h>
#include <v8.h>
#include <stdint.h>
#include <vector>

#include "crypto/verus_hash.h"
#include "sodium.h"

using namespace v8;

CVerusHash* vh;
CVerusHashV2* vh2;
CVerusHashV2* vh2b1;
CVerusHashV2* vh2b2;

bool initialized = false;

void initialize()
{
    if (!initialized)
    {
        CVerusHash::init();
        CVerusHashV2::init();
    }
    
    vh = new CVerusHash();
    vh2 = new CVerusHashV2(SOLUTION_VERUSHHASH_V2);
    vh2b1 = new CVerusHashV2(SOLUTION_VERUSHHASH_V2_1);
	vh2b2 = new CVerusHashV2(SOLUTION_VERUSHHASH_V2_2);
    
    initialized = true;
}

void verusInit(const v8::FunctionCallbackInfo<Value>& args) {
    initialize();
    args.GetReturnValue().Set(args.This());
}

void verusHash(const v8::FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    if (args.Length() < 1) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments"))
        );
        return;
    }
    MaybeLocal<Object> maybeBuffer = Nan::To<v8::Object>(args[0]);
    Local<Object> buffer;
    if (maybeBuffer.ToLocal(&buffer) != true) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }
    if(!node::Buffer::HasInstance(buffer)) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }

    const char *buff = node::Buffer::Data(buffer);

    char *result = new char[32];
    
    if (initialized == false) {
        initialize();
    }
    verus_hash(result, buff, node::Buffer::Length(buffer));
    args.GetReturnValue().Set(Nan::NewBuffer(result, 32).ToLocalChecked());
}

void verusHashV2(const v8::FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    if (args.Length() < 1) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments"))
        );
        return;
    }
    MaybeLocal<Object> maybeBuffer = Nan::To<v8::Object>(args[0]);
    Local<Object> buffer;    
    if (maybeBuffer.ToLocal(&buffer) != true) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }
    if(!node::Buffer::HasInstance(buffer)) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }

    const char *buff = node::Buffer::Data(buffer);

    char *result = new char[32];
    
    if (initialized == false) {
        initialize();
    }

    vh2->Reset();
    vh2->Write((const unsigned char *)buff, node::Buffer::Length(buffer));
    vh2->Finalize((unsigned char *)result);
    args.GetReturnValue().Set(Nan::NewBuffer(result, 32).ToLocalChecked());
}

void verusHashV2b(const v8::FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    if (args.Length() < 1) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments"))
        );
        return;
    }
    MaybeLocal<Object> maybeBuffer = Nan::To<v8::Object>(args[0]);
    Local<Object> buffer;    
    if (maybeBuffer.ToLocal(&buffer) != true) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }
    if(!node::Buffer::HasInstance(buffer)) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }

    const char *buff = node::Buffer::Data(buffer);

    char *result = new char[32];
    
    if (initialized == false) {
        initialize();
    }

    vh2->Reset();
    vh2->Write((const unsigned char *)buff, node::Buffer::Length(buffer));
    vh2->Finalize2b((unsigned char *)result);
    args.GetReturnValue().Set(Nan::NewBuffer(result, 32).ToLocalChecked());
}

void verusHashV2b1(const v8::FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    if (args.Length() < 1) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments"))
        );
        return;
    }
    MaybeLocal<Object> maybeBuffer = Nan::To<v8::Object>(args[0]);
    Local<Object> buffer;    
    if (maybeBuffer.ToLocal(&buffer) != true) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }
    if(!node::Buffer::HasInstance(buffer)) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }

    const char *buff = node::Buffer::Data(buffer);

    char *result = new char[32];
    
    if (initialized == false) {
        initialize();
    }

    vh2b1->Reset();
    vh2b1->Write((const unsigned char *)buff, node::Buffer::Length(buffer));
    vh2b1->Finalize2b((unsigned char *)result);
    args.GetReturnValue().Set(Nan::NewBuffer(result, 32).ToLocalChecked());
}

const unsigned char BLAKE2Bpersonal[crypto_generichash_blake2b_PERSONALBYTES] = { 'V','e','r','u','s','D','e','f','a','u','l','t','H','a','s','h' };
uint256 blake2b_hash(unsigned char* data, unsigned long long length)
{
    const unsigned char* personal = BLAKE2Bpersonal;
    crypto_generichash_blake2b_state state;
    uint256 result;
    if (crypto_generichash_blake2b_init_salt_personal(
        &state,
        NULL, 0, // No key.
        32,
        NULL,    // No salt.
        personal) == 0) {
        crypto_generichash_blake2b_update(&state, data, length);
        if (crypto_generichash_blake2b_final(&state, reinterpret_cast<unsigned char*>(&result), crypto_generichash_blake2b_BYTES) == 0) {
            return result;
        }
    }
    result.SetNull();
    return result;
}

void verusHashV2b2(const v8::FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    if (args.Length() < 1) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments"))
        );
        return;
    }
    MaybeLocal<Object> maybeBuffer = Nan::To<v8::Object>(args[0]);
    Local<Object> buffer;    
    if (maybeBuffer.ToLocal(&buffer) != true) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }
    if(!node::Buffer::HasInstance(buffer)) {
        isolate->ThrowException(
            Exception::TypeError(String::NewFromUtf8(isolate, "Invalid buffer objects."))
        );
        return;
    }

    char *buff = node::Buffer::Data(buffer);
    char* result = new char[32];
    
    if (initialized == false) {
        initialize();
    }

    // detect pbaas, validate and clear non-canonical data if needed
    char* solution = (buff + 140 + 3);
    unsigned int sol_ver = ((solution[0]) + (solution[1] << 8) + (solution[2] << 16) + (solution[3] << 24));
    if (sol_ver > 6) {
        //const uint8_t descrBits = solution[4];
        const uint8_t numPBaaSHeaders = solution[5];
        //const uint16_t extraSpace = solution[6] | ((uint16_t)(solution[7]) << 8);
        const uint32_t soln_header_size = 4 + 1 + 1 + 2 + 32 + 32; // version, descr, numPBaas, extraSpace, hashPrevMMRroot, hashBlockMMRroot
        const uint32_t soln_pbaas_cid_size = 20;   // hash160
        const uint32_t soln_pbaas_prehash_sz = 32; // pre header hash blake2b
        // if pbaas headers present
        if (numPBaaSHeaders > 0) {
            unsigned char preHeader[32 + 32 + 32 + 32 + 4 + 32 + 32] = { 0, };

            // copy non-canonical items from block header
            memcpy(&preHeader[0], buff + 4, 32);           // hashPrevBlock
            memcpy(&preHeader[32], buff + 4 + 32, 32);      // hashMerkleRoot
            memcpy(&preHeader[64], buff + 4 + 32 + 32, 32); // hashFinalSaplingRoot
            memcpy(&preHeader[96], buff + 4 + 32 + 32 + 32 + 4 + 4, 32); // nNonce (if nonce changes must update preHeaderHash in solution)
            memcpy(&preHeader[128], buff + 4 + 32 + 32 + 32 + 4, 4); // nbits
            memcpy(&preHeader[132], solution + 8, 32 + 32);  // hashPrevMMRRoot, hashPrevMMRRoot

            // detect if merged mining is present and clear non-canonical data (if needed)
            int matched_zeros = 0;
            for (int i = 0; i < sizeof(preHeader); i++) {
                if (preHeader[i] == 0) { matched_zeros++; }
            }

            // if the data has already been cleared of non-canonical data, just continue along
            if (matched_zeros != sizeof(preHeader)) {
                // detect merged mining by looking for preHeaderHash (blake2b) in first pbaas chain definition
                int matched_hashes = 0;
                uint256 preHeaderHash = blake2b_hash(&preHeader[0], sizeof(preHeader));
                if (!preHeaderHash.IsNull()) {
                    if (memcmp((unsigned char*)&preHeaderHash,
                        &solution[soln_header_size + soln_pbaas_cid_size],
                        soln_pbaas_prehash_sz) == 0) {
                        matched_hashes++;
                    }
                }
                // clear non-canonical data for pbaas merge mining
                if (matched_hashes > 0) {
                    memset(buff + 4, 0, 32 + 32 + 32);              // hashPrevBlock, hashMerkleRoot, hashFinalSaplingRoot
                    memset(buff + 4 + 32 + 32 + 32 + 4, 0, 4);      // nBits
                    memset(buff + 4 + 32 + 32 + 32 + 4 + 4, 0, 32); // nNonce
                    memset(solution + 8, 0, 32 + 32);               // hashPrevMMRRoot, hashBlockMMRRoot
                    //printf("info: merged mining %d chains, clearing non-canonical data on hash found\n", numPBaaSHeaders);
                } else {
                    // invalid share, pbaas activated must be pbaas mining capatible
                    memset(result, 0xff, 32);
                    args.GetReturnValue().Set(Nan::NewBuffer(result, 32).ToLocalChecked());
                    return;
                }
            } else {
                //printf("info: merged mining %d chains, non-canonical data pre-cleared\n", numPBaaSHeaders);
            }
        }
    }

    vh2b2->Reset();
    vh2b2->Write((const unsigned char *)buff, node::Buffer::Length(buffer));
    vh2b2->Finalize2b((unsigned char *)result);

    args.GetReturnValue().Set(Nan::NewBuffer(result, 32).ToLocalChecked());
}

void Init(Handle<Object> exports) {
  NODE_SET_METHOD(exports, "init", verusInit);

  NODE_SET_METHOD(exports, "hash", verusHash);          //VerusHash V1
  NODE_SET_METHOD(exports, "hash2", verusHashV2);       //VerusHash V2
  NODE_SET_METHOD(exports, "hash2b", verusHashV2b);     //VerusHash V2B
  NODE_SET_METHOD(exports, "hash2b1", verusHashV2b1);   //VerusHash V2B1
  NODE_SET_METHOD(exports, "hash2b2", verusHashV2b2);   //VerusHash V2B2
}

NODE_MODULE(verushash, Init)
