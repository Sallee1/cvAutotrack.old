using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace cvAutoTrackCSharpImpl
{
    internal class cvAutoTrack
    {
        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(String path);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetProcAddress(IntPtr lib, String funcName);

        [DllImport("kernel32.dll")]
        private static extern bool FreeLibrary(IntPtr lib);

        public delegate void CallBack_SetServerCallback(byte[] buff, int size);

        public delegate long CallBack_SetCaptureHandleCallback();

        public delegate void CallBack_SetScreenSourceCallback(byte[] image_encode_data, ref int image_data_size);

        public delegate void CallBack_SetScreenSourceCallbackEx(byte[] image_data, ref int image_width, ref int image_height, ref int image_channels);

        public delegate void CallBack_SetScreenClientRectCallback(ref int x, ref int y, ref int width, ref int height);

        public delegate bool Call_DebugLoadMapImagePath(byte[] path);

        public delegate bool Call_InitResource();

        public delegate bool Call_UnInitResource();

        public delegate bool Call_SetCacheConfig(byte[] config_file, byte[] blocks_dir, byte[] config, int config_size);

        public delegate bool Call_SetCoreCachePath(byte[] path);

        public delegate bool Call_GetCoreCachePath(byte[] path_buff, int buff_size);

        public delegate bool Call_StartServer();

        public delegate bool Call_StopServer();

        public delegate bool Call_SetServerInterval(int interval_ms);

        public delegate bool Call_SetServerCallback(CallBack_SetServerCallback callback);

        public delegate bool Call_SetDisableFileLog();

        public delegate bool Call_SetEnableFileLog();

        public delegate bool Call_SetLogFilePath(byte[] path);

        public delegate bool Call_SetLogFileName(byte[] path);

        public delegate bool Call_SetUseBitbltCaptureMode();

        public delegate bool Call_SetUseGraphicsCaptureMode();

        public delegate bool Call_SetUseDwmCaptureMode();

        public delegate bool Call_SetUseLocalPictureMode();

        public delegate bool Call_SetUseLocalVideoMode();

        public delegate bool Call_SetCaptureHandle(long handle);

        public delegate bool Call_SetCaptureHandleCallback(CallBack_SetCaptureHandleCallback callback);

        public delegate bool Call_SetScreenSourceCallback(CallBack_SetScreenSourceCallback callback);

        public delegate bool Call_SetScreenSourceCallbackEx(CallBack_SetScreenSourceCallbackEx callback);

        public delegate bool Call_SetScreenSourceImage(byte[] image_encode_data, int image_data_size);

        public delegate bool Call_SetScreenSourceImageEx(byte[] image_data, int image_width, int image_height, int height);

        public delegate bool Call_SetScreenClientRectCallback(CallBack_SetScreenClientRectCallback callback);

        public delegate bool Call_SetTrackCachePath(byte[] path);

        public delegate bool Call_SetTrackCacheName(byte[] path);

        public delegate bool Call_SetWorldCenter(double x, double y);

        public delegate bool Call_SetWorldScale(double scale);

        public delegate bool Call_GetTransformOfMap(ref double x, ref double y, ref double a, ref int map_id);

        public delegate bool Call_GetPositionOfMap(ref double x, ref double y, ref int map_id);

        public delegate bool Call_GetDirection(ref double a);

        public delegate bool Call_GetRotation(ref double a);

        public delegate bool Call_GetUID(ref int uid);

        public delegate bool Call_GetAllInfo(ref double x, ref double y, ref int map_id, ref double a, ref double r, ref int uid);

        public delegate bool Call_DebugCapture();

        public delegate bool Call_DebugCapturePath(byte[] path);

        public delegate int Call_GetLastErr();

        public delegate int Call_GetLastErrMsg(byte[] buff, int size);

        public delegate int Call_GetLastErrJson(byte[] buff, int size);

        public delegate bool Call_GetCompileVersion(byte[] buff, int size);

        public delegate bool Call_GetCompileTime(byte[] buff, int size);

        public delegate bool Call_GetCoreModulePath(byte[] buff, int size);

        public delegate bool Call_LoadDependModuleFromPath(byte[] buff, int size);

        [StructLayout(LayoutKind.Sequential)]
        public struct cvAutoTrackContextV1
        {
            public Call_DebugLoadMapImagePath DebugLoadMapImagePath;
            public Call_InitResource InitResource;
            public Call_UnInitResource UnInitResource;
            public Call_SetCacheConfig SetCacheConfig;
            public Call_SetCoreCachePath SetCoreCachePath;
            public Call_GetCoreCachePath GetCoreCachePath;
            public Call_StartServer StartServer;
            public Call_StopServer StopServer;
            public Call_SetServerInterval SetServerInterval;
            public Call_SetServerCallback SetServerCallback;
            public Call_SetDisableFileLog SetDisableFileLog;
            public Call_SetEnableFileLog SetEnableFileLog;
            public Call_SetLogFilePath SetLogFilePath;
            public Call_SetLogFileName SetLogFileName;
            public Call_SetUseBitbltCaptureMode SetUseBitbltCaptureMode;
            public Call_SetUseGraphicsCaptureMode SetUseGraphicsCaptureMode;
            public Call_SetUseDwmCaptureMode SetUseDwmCaptureMode;
            public Call_SetUseLocalPictureMode SetUseLocalPictureMode;
            public Call_SetUseLocalVideoMode SetUseLocalVideoMode;
            public Call_SetCaptureHandle SetCaptureHandle;
            public Call_SetCaptureHandleCallback SetCaptureHandleCallback;
            public Call_SetScreenSourceCallback SetScreenSourceCallback;
            public Call_SetScreenSourceCallbackEx SetScreenSourceCallbackEx;
            public Call_SetScreenSourceImage SetScreenSourceImage;
            public Call_SetScreenSourceImageEx SetScreenSourceImageEx;
            public Call_SetScreenClientRectCallback SetScreenClientRectCallback;
            public Call_SetTrackCachePath SetTrackCachePath;
            public Call_SetTrackCacheName SetTrackCacheName;
            public Call_SetWorldCenter SetWorldCenter;
            public Call_SetWorldScale SetWorldScale;
            public Call_GetTransformOfMap GetTransformOfMap;
            public Call_GetPositionOfMap GetPositionOfMap;
            public Call_GetDirection GetDirection;
            public Call_GetRotation GetRotation;
            public Call_GetUID GetUID;
            public Call_GetAllInfo GetAllInfo;
            public Call_DebugCapture DebugCapture;
            public Call_DebugCapturePath DebugCapturePath;
            public Call_GetLastErr GetLastErr;
            public Call_GetLastErrMsg GetLastErrMsg;
            public Call_GetLastErrJson GetLastErrJson;
            public Call_GetCompileVersion GetCompileVersion;
            public Call_GetCompileTime GetCompileTime;
            public Call_GetCoreModulePath GetCoreModulePath;
        }

        private delegate IntPtr Call_create_cvAutoTrack_context_v1();

        private delegate void Call_destroy_cvAutoTrack_context_v1(IntPtr ctx);

        private IntPtr Ptr_create_cvAutoTrack_context_v1;
        private IntPtr Ptr_destroy_cvAutoTrack_context_v1;

        private IntPtr lib = IntPtr.Zero;

        public bool init_dll()
        {
            lib = LoadLibrary("cvAutoTrack.dll");
            if (lib == IntPtr.Zero)
            {
                IsLoadLibrary = false;
                return false;
            }
            Ptr_create_cvAutoTrack_context_v1 = GetProcAddress(lib, "create_cvAutoTrack_context_v1");
            Ptr_destroy_cvAutoTrack_context_v1 = GetProcAddress(lib, "destroy_cvAutoTrack_context_v1");

            if (Ptr_create_cvAutoTrack_context_v1 == IntPtr.Zero)
            {
                IsLoadLibrary = false;
            }
            if (Ptr_destroy_cvAutoTrack_context_v1 == IntPtr.Zero)
            {
                IsLoadLibrary = false;
            }

            IsLoadLibrary = true;
            return IsLoadLibrary;
        }

        public void uninit_dll()
        {
            if (IsLoadLibrary)
            {
                FreeLibrary(lib);
                IsLoadLibrary = false;
            }
        }

        public bool IsLoadLibrary = false;

        public cvAutoTrack(bool is_auto_load = true)
        {
            if (is_auto_load)
            {
                init_dll();
            }
        }

        ~cvAutoTrack()
        {
            uninit_dll();
        }

        public cvAutoTrackContextV1 create_cvAutoTrack_context_v1()
        {
            Call_create_cvAutoTrack_context_v1 create_cvAutoTrack_context_v1_Dete = (Call_create_cvAutoTrack_context_v1)Marshal.GetDelegateForFunctionPointer(Ptr_create_cvAutoTrack_context_v1, typeof(Call_create_cvAutoTrack_context_v1));

            IntPtr ptr = create_cvAutoTrack_context_v1_Dete();
            if (ptr == IntPtr.Zero)
            {
                throw new NullReferenceException();
            }

            return Marshal.PtrToStructure<cvAutoTrackContextV1>(ptr);
        }

        public void destroy_cvAutoTrack_context_v1(cvAutoTrackContextV1 ctx)
        {
            Call_destroy_cvAutoTrack_context_v1 destroy_cvAutoTrack_context_v1_Dete = (Call_destroy_cvAutoTrack_context_v1)Marshal.GetDelegateForFunctionPointer(Ptr_destroy_cvAutoTrack_context_v1, typeof(Call_destroy_cvAutoTrack_context_v1));
            IntPtr ptr = IntPtr.Zero;
            Marshal.StructureToPtr<cvAutoTrackContextV1>(ctx, ptr, false);
            destroy_cvAutoTrack_context_v1_Dete(ptr);
        }
    }
}