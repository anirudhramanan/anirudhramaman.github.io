---
title: Why Flatbuffers ? 
date: 2016-09-14
tags: [android, flatbuffers]
---

[![Serialization Performance](https://img.youtube.com/vi/IwxIIUypnTE/0.jpg)](https://www.youtube.com/watch?v=IwxIIUypnTE)

Why Flatbuffers? Why not use some standard libraries such as GSON, or jackson. Is there any benefit of using flatbuffers over other libraries. This post answers it all.

[Flatbuffers](https://github.com/google/flatbuffers), by google is an efficient cross platform serialization library, which was originally targeted for games. Instead of jumping into the benefits of Flatbuffers, which you can anyway go through in their [official site](http://google.github.io/flatbuffers/), I will share some real performance numbers while compared to GSON on the demo app.

Here's the sample json which was used for the test

{% highlight groovy %}
{
  "event": [
    {
      "id": 1,
      "name": "Android Conference",
      "location": "Delhi",
      "date": "2016/04/05"
    },
    {
      "id": 2,
      "name": "iOS Conference",
      "location": "Bangalore",
      "date": "2016/05/01"
    },
    ...
    ...
}
{% endhighlight %}

And, here's the flat buffer schema for the same.

{% highlight groovy %}
table EventList {
    event : [Event];
}

table Event {
    id : long;
    name : string;
    location : string;
    date : string;
}

root_type EventList;
{% endhighlight %}

Using the flatc compiler, I generated the java classes for the same. You can find the instructions on how to use the compiler in the github repo [README](https://github.com/anirudhramanan/flatbuffers-android-demo) file.

Here are some metrics which I measuread using both :

* <b>Loading Time</b>

With gson, it took around 30 - 40 ms to parse the entire json file, whereas with flatbuffers it took 3 - 4 ms for the same. Reason being the data is stored in byte buffer format in the flatbuffer file, so there is no need for parsing or unpacking the file, thus resulting in lesser time accessing the data.

![Comparison](../content/images/2016/flatbuff/comparison.png)

* <b>Memory Allocation</b>

With flatbuffers, since only the buffer uses memory, there were lesser GC calls as compared to Gson (where a lot of small objects are created which needs GC calls to free up memory). As a result the CPU cycle calls were also less. Hence, using flatbuffers proved to be much memory efficient as compared to Gson.

### More articles on Flatbuffers

* Facebook : A nice read on how flatbuffers helped them speed up loading their posts. [Link](https://code.facebook.com/posts/872547912839369/improving-facebook-s-performance-on-android-with-flatbuffers/)

### Sample Project

You can head over to the Github [project](https://github.com/anirudhramanan/flatbuffers-android-demo) for more information. 

