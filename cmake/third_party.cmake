# OpenCV

set(OpenCV_DIR  "C:/Packages/opencv-static/staticlib"  CACHE PATH "OpenCV 预编译库路径")
find_package(OpenCV REQUIRED)

# Faiss
set(faiss_DIR "C:/Packages/faiss/share/faiss")
find_package(faiss REQUIRED)

# OpenSSL 导入目标
set(OPENSSL_LIBRARY "C:/Packages/openssl-static" CACHE PATH "OpenSSL 根目录")
if(NOT TARGET OpenSSL::Crypto)
  add_library(OpenSSL::Crypto STATIC IMPORTED)
  set_target_properties(OpenSSL::Crypto PROPERTIES
    IMPORTED_LOCATION "${OPENSSL_LIBRARY}/lib/libcrypto.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_LIBRARY}/include"
    INTERFACE_LINK_LIBRARIES "ws2_32.lib;crypt32.lib;advapi32.lib;user32.lib")
endif()
if(NOT TARGET OpenSSL::SSL)
  add_library(OpenSSL::SSL STATIC IMPORTED)
  set_target_properties(OpenSSL::SSL PROPERTIES
    IMPORTED_LOCATION "${OPENSSL_LIBRARY}/lib/libssl.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_LIBRARY}/include"
    INTERFACE_LINK_LIBRARIES OpenSSL::Crypto)
endif()

# cURL 导入目标
set(CURL_LABRARY "C:/Packages/curl-static" CACHE PATH "cURL 根目录")
if(NOT TARGET CURL::libcurl)
  add_library(CURL::libcurl STATIC IMPORTED)
  set_target_properties(CURL::libcurl PROPERTIES
    IMPORTED_LOCATION "${CURL_LABRARY}/lib/libcurl.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CURL_LABRARY}/include"
    INTERFACE_LINK_LIBRARIES
      "ws2_32.lib;iphlpapi.lib;bcrypt.lib;advapi32.lib;crypt32.lib"
  )
endif()

# nlohmann-json 纯头目标
set(NLOHMANN_JSON_DIR "C:/Packages/nlohmann_json" CACHE PATH "nlohmann-json 头文件目录")
if(NOT TARGET nlohmann_json::nlohmann_json)
  add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
  set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_DIR}/include")
endif()