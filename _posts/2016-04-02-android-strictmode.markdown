---
title: Strict Mode in Android
date: 2016-04-02
tags: [android, strictmode]
---

Not many of you may have heard about Strict Mode in Android. Added in Api Level 9, it is a developer tool which detects those part of codes that may crash due to some voilations, and brings them to your attention with an error code that is easy to understand.

The Strict mode can help you detect various voilations such as :

#### Disk IO on Main Thread

Catch all accidental disk operations taking place on the main thread.

#### Network Access on Main Thread

Catch all kind of network operations that are taking place in the main thread.

#### Leaked Objects

Detect all kind of leaked objects such as sql, cursors and much more.

StrictMode is most commonly used to catch accidental disk or network access on the application's main thread, where UI operations are received and animations take place. Keeping disk and network operations off the main thread makes for much smoother, more responsive applications.

[Strict Mode Official Document](http://developer.android.com/reference/android/os/StrictMode.html)

### How to enable Strict Mode?

Enabling strict mode is just a piece of cake. Just few lines of code in your application class, and its over.

- NOTE : Strict Mode should only be used in DEBUG mode.

{% highlight groovy %}
public void onCreate() {

    if (DEVELOPER_MODE) {
        StrictMode.setThreadPolicy(new StrictMode.ThreadPolicy.Builder()
        .detectDiskReads()
        .detectDiskWrites()
        .detectNetwork()   // or .detectAll() for all detectable problems
        .penaltyLog()
        .build());

        StrictMode.setVmPolicy(new StrictMode.VmPolicy.Builder()
        .detectLeakedSqlLiteObjects()
        .detectLeakedClosableObjects()
        .penaltyLog()
        .penaltyDeath()
        .build());
    }
    super.onCreate();

}
{% endhighlight %}
