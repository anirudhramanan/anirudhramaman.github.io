---
title: Choosing the right Scheduler API
date: 2016-10-15
tags: [android]
image: "./strictmode.jpg"
---

Most of the apps uses schedulers to schedule task that are executed outside of the life cycle i.e even when the app isn't running. Tasks such as checking the server for updates, pushing analytics events etc. There are several API's for scheduling tasks, but not all fits the every use-case. 

Lets go through all the API's and decide the best one for each use case.

## Job Scheduler 

[JobScheduler](https://developer.android.com/reference/android/app/job/JobScheduler.html) was added in <b>API Level 21</b>, this is used to schedule jobs in the application's own process. It helps scheduling jobs in most efficient way, taking into account the battery status, phone charging status, and also the network status i.e if the network is metered or unmetered.

<h4>Advantages of using Job Scheduler ?</h4>
  1. Tasks that should be executed only when the device is in charging mode.
  2. Tasks that are executed periodically, e.g pushing analytical event daily.
  3. Tasks that should be executed only on WIFI.

<h4>How to use Job Scheduler?</h4>

<b>JobService</b> : JobService class is used for implementing the onStartJob() and onStopJob(). This has to be registered with your manifest file.

  * <b>onStartJob()</b> : is the method where we define the callback code for the jobs, and it is executed on the Main thread. If the job can be done on the main thread, return false, else if it is a long-running job and has to be done async, return true.

  * <b>onStopJob()</b> : is called when the job stops, maybe due to some paramater mismatch such as when user unplugs the device from charging. Return true if you want to reschedule the job incase it fails due to some reason.

  {% highlight groovy %}
    public class CustomJobService extends JobService {
      private static final String TAG = "CustomJobService";

      @Override
      public boolean onStartJob(JobParameters params) {
        Log.i(TAG, "on start job: " + params.getJobId());
        return true;
      }

      @Override
      public boolean onStopJob(JobParameters params) {
        Log.i(TAG, "on stop job: " + params.getJobId());
        return true;
        }
    }
  {% endhighlight %}

<b>JobInfo</b> : JobInfo is used for creating the job with specified parameters for scheduling. 

  {% highlight groovy %}
    ComponentName serviceComponent = new ComponentName(context, CustomJobService.class);
      
    //create the jobinfo builder object with id and component name
    JobInfo.Builder builder = new JobInfo.Builder(1, serviceComponent);
    
    //the job is delayed by 5 sec. This should not be used for periodic jobs, will throw IllegalArgumentException
    builder.setMinimumLatency(5 * 1000);

    //requires unmetered network
    builder.setRequiredNetworkType(JobInfo.NETWORK_TYPE_UNMETERED); 
    
    //the device should be in idle state
    builder.setRequiresDeviceIdle(true); 

    //the device should be in charging mode
    builder.setRequiresCharging(true);

    //use the JobScheduler API to schedule the job
    JobScheduler jobScheduler = context.getSystemService(JobScheduler.class);
    jobScheduler.schedule(builder.build());
  {% endhighlight %}

NOTE : JobScheduler is compatible only with API Level 21 and above. For backport, GCM Network Manager is used.

## GCM (Google Cloud Messaging) Network Manager

[GCM Network Manager](https://developers.google.com/android/reference/com/google/android/gms/gcm/GcmNetworkManager) is used as a backport for JobScheduler for below API level 21. It internally uses JobScheduler for API 21 and above. 

This uses the Google Play Services to help schedule and batch jobs across the system. Tasks Builder(replacement of JobInfo) are used to define an one-time or a periodic task, and the system takes care of optimizing the battery usage by automatically batching the tasks.

<h4>How to use GCM Network Manager?</h4>

<b>GcmTaskService</b> : This is used to receive the notification from the scheduler that a task is ready to be executed. We will implement this service and subclass the onRunTask() method to do our job. This has to be registered with your manifest file.

  {% highlight groovy %}
    public class CustomGcmService extends GcmTaskService {
         @Override
         public int onRunTask(TaskParams params) {
             // Do some work
             return GcmNetworkManager.RESULT_SUCCESS;
         }
     }
  {% endhighlight %}

<b>Tasks</b> : There are two types of tasks, OneoffTask and PeriodicTask.

* <b>OneoffTask</b> : A task that is executed once, at some point within the specified window. 

  {% highlight groovy %}
    OneoffTask oneofftask = new OneoffTask.Builder()
                    .setService(CustomGcmService.class)
                    // Schedule a task to occur between five and ten minutes from now:
                    .setExecutionWindow(5 * DateUtil.MINUTE_IN_SECONDS, 10 * DateUtil.MINUTE_IN_SECONDS)
                    //requires charging
                    .setRequiresCharging(true)
                    //task should be persisted across reboots, must have RECEIVE_BOOT_COMPLETED permission
                    .setPersisted(true)
                    .build();

  GcmNetworkManager.getInstance(this).schedule(oneofftask);
  {% endhighlight %}

* <b>PeriodicTask</b> : Task that is executed at the specified interval, without needing to be rescheduled.

  {% highlight groovy %}
    PeriodicTask periodictask = new PeriodicTask.Builder()
                    .setService(CustomGcmService.class)
                    //tasks to recur at specified intervals, in seconds.
                    .setPeriod(5000)
                    //requires charging
                    .setRequiresCharging(true)
                    //task should be persisted across reboots, must have RECEIVE_BOOT_COMPLETED permission
                    .setPersisted(true)
                    .build();

  GcmNetworkManager.getInstance(this).schedule(periodictask);
  {% endhighlight %}

GcmNetworkManager, and JobScheduler fits the use case when the job has to be repeated or one-off, while keeping battery life in mind. If some kind of precision is required in the timing, AlarmManager should be used. 

Eg : An alarm app should use AlarmManager, instead of GcmNetworkManager, or JobScheduler.