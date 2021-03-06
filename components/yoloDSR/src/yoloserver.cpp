// **********************************************************************
//
// Copyright (c) 2003-2013 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.5.1
//
// <auto-generated>
//
// Generated from file `yoloserver.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#include <yoloserver.h>
#include <Ice/LocalException.h>
#include <Ice/ObjectFactory.h>
#include <Ice/BasicStream.h>
#include <Ice/Object.h>
#include <IceUtil/Iterator.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 305
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 1
#       error Ice patch level mismatch!
#   endif
#endif

namespace
{

const ::std::string __RoboCompYoloServer__YoloServer__addImage_name = "addImage";

const ::std::string __RoboCompYoloServer__YoloServer__getData_name = "getData";

}

namespace Ice
{
}
::IceProxy::Ice::Object* ::IceProxy::RoboCompYoloServer::upCast(::IceProxy::RoboCompYoloServer::YoloServer* p) { return p; }

void
::IceProxy::RoboCompYoloServer::__read(::IceInternal::BasicStream* __is, ::IceInternal::ProxyHandle< ::IceProxy::RoboCompYoloServer::YoloServer>& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::RoboCompYoloServer::YoloServer;
        v->__copyFrom(proxy);
    }
}

::Ice::Int
IceProxy::RoboCompYoloServer::YoloServer::addImage(const ::RoboCompYoloServer::Image& img, const ::Ice::Context* __ctx)
{
    ::IceInternal::InvocationObserver __observer(this, __RoboCompYoloServer__YoloServer__addImage_name, __ctx);
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__RoboCompYoloServer__YoloServer__addImage_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::RoboCompYoloServer::YoloServer* __del = dynamic_cast< ::IceDelegate::RoboCompYoloServer::YoloServer*>(__delBase.get());
            return __del->addImage(img, __ctx, __observer);
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, __observer);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, true, __cnt, __observer);
        }
    }
}

::Ice::AsyncResultPtr
IceProxy::RoboCompYoloServer::YoloServer::begin_addImage(const ::RoboCompYoloServer::Image& img, const ::Ice::Context* __ctx, const ::IceInternal::CallbackBasePtr& __del, const ::Ice::LocalObjectPtr& __cookie)
{
    __checkAsyncTwowayOnly(__RoboCompYoloServer__YoloServer__addImage_name);
    ::IceInternal::OutgoingAsyncPtr __result = new ::IceInternal::OutgoingAsync(this, __RoboCompYoloServer__YoloServer__addImage_name, __del, __cookie);
    try
    {
        __result->__prepare(__RoboCompYoloServer__YoloServer__addImage_name, ::Ice::Normal, __ctx);
        ::IceInternal::BasicStream* __os = __result->__startWriteParams(::Ice::DefaultFormat);
        __os->write(img);
        __result->__endWriteParams();
        __result->__send(true);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __result->__exceptionAsync(__ex);
    }
    return __result;
}

::Ice::Int
IceProxy::RoboCompYoloServer::YoloServer::end_addImage(const ::Ice::AsyncResultPtr& __result)
{
    ::Ice::AsyncResult::__check(__result, this, __RoboCompYoloServer__YoloServer__addImage_name);
    ::Ice::Int __ret;
    bool __ok = __result->__wait();
    try
    {
        if(!__ok)
        {
            try
            {
                __result->__throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                throw ::Ice::UnknownUserException(__FILE__, __LINE__, __ex.ice_name());
            }
        }
        ::IceInternal::BasicStream* __is = __result->__startReadParams();
        __is->read(__ret);
        __result->__endReadParams();
        return __ret;
    }
    catch(const ::Ice::LocalException& ex)
    {
        __result->__getObserver().failed(ex.ice_name());
        throw;
    }
}

::RoboCompYoloServer::Labels
IceProxy::RoboCompYoloServer::YoloServer::getData(::Ice::Int id, const ::Ice::Context* __ctx)
{
    ::IceInternal::InvocationObserver __observer(this, __RoboCompYoloServer__YoloServer__getData_name, __ctx);
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__RoboCompYoloServer__YoloServer__getData_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::RoboCompYoloServer::YoloServer* __del = dynamic_cast< ::IceDelegate::RoboCompYoloServer::YoloServer*>(__delBase.get());
            return __del->getData(id, __ctx, __observer);
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, __observer);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, true, __cnt, __observer);
        }
    }
}

::Ice::AsyncResultPtr
IceProxy::RoboCompYoloServer::YoloServer::begin_getData(::Ice::Int id, const ::Ice::Context* __ctx, const ::IceInternal::CallbackBasePtr& __del, const ::Ice::LocalObjectPtr& __cookie)
{
    __checkAsyncTwowayOnly(__RoboCompYoloServer__YoloServer__getData_name);
    ::IceInternal::OutgoingAsyncPtr __result = new ::IceInternal::OutgoingAsync(this, __RoboCompYoloServer__YoloServer__getData_name, __del, __cookie);
    try
    {
        __result->__prepare(__RoboCompYoloServer__YoloServer__getData_name, ::Ice::Normal, __ctx);
        ::IceInternal::BasicStream* __os = __result->__startWriteParams(::Ice::DefaultFormat);
        __os->write(id);
        __result->__endWriteParams();
        __result->__send(true);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __result->__exceptionAsync(__ex);
    }
    return __result;
}

::RoboCompYoloServer::Labels
IceProxy::RoboCompYoloServer::YoloServer::end_getData(const ::Ice::AsyncResultPtr& __result)
{
    ::Ice::AsyncResult::__check(__result, this, __RoboCompYoloServer__YoloServer__getData_name);
    ::RoboCompYoloServer::Labels __ret;
    bool __ok = __result->__wait();
    try
    {
        if(!__ok)
        {
            try
            {
                __result->__throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                throw ::Ice::UnknownUserException(__FILE__, __LINE__, __ex.ice_name());
            }
        }
        ::IceInternal::BasicStream* __is = __result->__startReadParams();
        __is->read(__ret);
        __result->__endReadParams();
        return __ret;
    }
    catch(const ::Ice::LocalException& ex)
    {
        __result->__getObserver().failed(ex.ice_name());
        throw;
    }
}

const ::std::string&
IceProxy::RoboCompYoloServer::YoloServer::ice_staticId()
{
    return ::RoboCompYoloServer::YoloServer::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::RoboCompYoloServer::YoloServer::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::RoboCompYoloServer::YoloServer);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::RoboCompYoloServer::YoloServer::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::RoboCompYoloServer::YoloServer);
}

::IceProxy::Ice::Object*
IceProxy::RoboCompYoloServer::YoloServer::__newInstance() const
{
    return new YoloServer;
}

::Ice::Int
IceDelegateM::RoboCompYoloServer::YoloServer::addImage(const ::RoboCompYoloServer::Image& img, const ::Ice::Context* __context, ::IceInternal::InvocationObserver& __observer)
{
    ::IceInternal::Outgoing __og(__handler.get(), __RoboCompYoloServer__YoloServer__addImage_name, ::Ice::Normal, __context, __observer);
    try
    {
        ::IceInternal::BasicStream* __os = __og.startWriteParams(::Ice::DefaultFormat);
        __os->write(img);
        __og.endWriteParams();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    ::Ice::Int __ret;
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        ::IceInternal::BasicStream* __is = __og.startReadParams();
        __is->read(__ret);
        __og.endReadParams();
        return __ret;
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

::RoboCompYoloServer::Labels
IceDelegateM::RoboCompYoloServer::YoloServer::getData(::Ice::Int id, const ::Ice::Context* __context, ::IceInternal::InvocationObserver& __observer)
{
    ::IceInternal::Outgoing __og(__handler.get(), __RoboCompYoloServer__YoloServer__getData_name, ::Ice::Normal, __context, __observer);
    try
    {
        ::IceInternal::BasicStream* __os = __og.startWriteParams(::Ice::DefaultFormat);
        __os->write(id);
        __og.endWriteParams();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    ::RoboCompYoloServer::Labels __ret;
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        ::IceInternal::BasicStream* __is = __og.startReadParams();
        __is->read(__ret);
        __og.endReadParams();
        return __ret;
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

::Ice::Int
IceDelegateD::RoboCompYoloServer::YoloServer::addImage(const ::RoboCompYoloServer::Image& img, const ::Ice::Context* __context, ::IceInternal::InvocationObserver&)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Int& __result, const ::RoboCompYoloServer::Image& __p_img, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _result(__result),
            _m_img(__p_img)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::RoboCompYoloServer::YoloServer* servant = dynamic_cast< ::RoboCompYoloServer::YoloServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            _result = servant->addImage(_m_img, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Int& _result;
        const ::RoboCompYoloServer::Image& _m_img;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __RoboCompYoloServer__YoloServer__addImage_name, ::Ice::Normal, __context);
    ::Ice::Int __result;
    try
    {
        _DirectI __direct(__result, img, __current);
        try
        {
            __direct.getServant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
    return __result;
}

::RoboCompYoloServer::Labels
IceDelegateD::RoboCompYoloServer::YoloServer::getData(::Ice::Int id, const ::Ice::Context* __context, ::IceInternal::InvocationObserver&)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::RoboCompYoloServer::Labels& __result, ::Ice::Int __p_id, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _result(__result),
            _m_id(__p_id)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::RoboCompYoloServer::YoloServer* servant = dynamic_cast< ::RoboCompYoloServer::YoloServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            _result = servant->getData(_m_id, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::RoboCompYoloServer::Labels& _result;
        ::Ice::Int _m_id;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __RoboCompYoloServer__YoloServer__getData_name, ::Ice::Normal, __context);
    ::RoboCompYoloServer::Labels __result;
    try
    {
        _DirectI __direct(__result, id, __current);
        try
        {
            __direct.getServant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
    return __result;
}

::Ice::Object* RoboCompYoloServer::upCast(::RoboCompYoloServer::YoloServer* p) { return p; }

namespace
{
const ::std::string __RoboCompYoloServer__YoloServer_ids[2] =
{
    "::Ice::Object",
    "::RoboCompYoloServer::YoloServer"
};

}

bool
RoboCompYoloServer::YoloServer::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__RoboCompYoloServer__YoloServer_ids, __RoboCompYoloServer__YoloServer_ids + 2, _s);
}

::std::vector< ::std::string>
RoboCompYoloServer::YoloServer::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__RoboCompYoloServer__YoloServer_ids[0], &__RoboCompYoloServer__YoloServer_ids[2]);
}

const ::std::string&
RoboCompYoloServer::YoloServer::ice_id(const ::Ice::Current&) const
{
    return __RoboCompYoloServer__YoloServer_ids[1];
}

const ::std::string&
RoboCompYoloServer::YoloServer::ice_staticId()
{
    return __RoboCompYoloServer__YoloServer_ids[1];
}

::Ice::DispatchStatus
RoboCompYoloServer::YoloServer::___addImage(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.startReadParams();
    ::RoboCompYoloServer::Image img;
    __is->read(img);
    __inS.endReadParams();
    ::Ice::Int __ret = addImage(img, __current);
    ::IceInternal::BasicStream* __os = __inS.__startWriteParams(::Ice::DefaultFormat);
    __os->write(__ret);
    __inS.__endWriteParams(true);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
RoboCompYoloServer::YoloServer::___getData(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.startReadParams();
    ::Ice::Int id;
    __is->read(id);
    __inS.endReadParams();
    ::RoboCompYoloServer::Labels __ret = getData(id, __current);
    ::IceInternal::BasicStream* __os = __inS.__startWriteParams(::Ice::DefaultFormat);
    __os->write(__ret);
    __inS.__endWriteParams(true);
    return ::Ice::DispatchOK;
}

namespace
{
const ::std::string __RoboCompYoloServer__YoloServer_all[] =
{
    "addImage",
    "getData",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

}

::Ice::DispatchStatus
RoboCompYoloServer::YoloServer::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< const ::std::string*, const ::std::string*> r = ::std::equal_range(__RoboCompYoloServer__YoloServer_all, __RoboCompYoloServer__YoloServer_all + 6, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __RoboCompYoloServer__YoloServer_all)
    {
        case 0:
        {
            return ___addImage(in, current);
        }
        case 1:
        {
            return ___getData(in, current);
        }
        case 2:
        {
            return ___ice_id(in, current);
        }
        case 3:
        {
            return ___ice_ids(in, current);
        }
        case 4:
        {
            return ___ice_isA(in, current);
        }
        case 5:
        {
            return ___ice_ping(in, current);
        }
    }

    assert(false);
    throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
}

void
RoboCompYoloServer::YoloServer::__writeImpl(::IceInternal::BasicStream* __os) const
{
    __os->startWriteSlice(ice_staticId(), -1, true);
    __os->endWriteSlice();
}

void
RoboCompYoloServer::YoloServer::__readImpl(::IceInternal::BasicStream* __is)
{
    __is->startReadSlice();
    __is->endReadSlice();
}

void 
RoboCompYoloServer::__patch(YoloServerPtr& handle, const ::Ice::ObjectPtr& v)
{
    handle = ::RoboCompYoloServer::YoloServerPtr::dynamicCast(v);
    if(v && !handle)
    {
        IceInternal::Ex::throwUOE(::RoboCompYoloServer::YoloServer::ice_staticId(), v);
    }
}
