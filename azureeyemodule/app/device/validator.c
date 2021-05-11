// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <pthread.h>
#include <libudev.h>
#include <libusb-1.0/libusb.h>
#include "validator.h"

#define AUTH_LOG
#define MAX_PACKAGE_SIZE 5120

// Epoll_wait timeout in ms
#define NETWORK_TIMEOUT 30000

static bool auth_pass = false;
static bool run = true;
static bool som_close_required = false;
static bool auth_pending = false;

static pthread_t validator_thread_id;
static char som_node[128] = {0};

// We should use lsusb -v to find the Endpoint addressvalues corresponding to the device.
static const unsigned char EP_IN_ADDRESS = 0x81;
static const unsigned char EP_OUT_ADDRESS = 0x01;

// Communication protocol between SoM and Validator
static const unsigned char StartAuthentication[2] = {0x77, 0x01};
static const unsigned char SendTLSDataToService[2] = {0x77, 0x02};
static const unsigned char ReceiveTLSDataFromService[2] = {0x77, 0x03};
static const unsigned char FailAuthentication[2] = {0x77, 0x04};
static const unsigned char PassAuthentication[2] = {0x77, 0x05};

// Azure Percept authentication service URL
static const char PublicServiceUrl[] = "auth.projectsantacruz.azure.net";
static const char TestServiceUrl[] = "dev.auth.projectsantacruz.azure.net";
static const char ServicePort[] = "443";

// Default device attribute
static const int Stm32CdcPid = 0x5740;
static const int Stm32CdcVid = 0x0483;
static const int Stm32CdcClass = 2;
static const int AzureEyeGen1Pid = 0x066F;
static const int AzureEyeGen1Version = 0x200;

// SoM board info
struct SOM_ID
{
    uint16_t vid;
    uint16_t pid;
};

static struct SOM_ID som_id;

// Structure stores the authentication operation required data
struct SOM_INFO
{
    libusb_device_handle *devh;
    int iofd;
    int epfd;
    unsigned char serial_data[MAX_PACKAGE_SIZE];
};

// Store the current opened device information
static struct SOM_INFO som_info;

// New net socket to connect server
static int new_net_connect(const char *host, const char *port)
{
    struct addrinfo hints, *addr_list, *cur;
    int fd = -1;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = 0;

    if (getaddrinfo(host, port, &hints, &addr_list) != 0)
    {
        return -1;
    }

    // Try the sockaddrs until a connection succeeds
    for (cur = addr_list; cur != NULL; cur = cur->ai_next)
    {
        fd = (int)socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
        if (fd < 0)
        {
            continue;
        }

        if (connect(fd, cur->ai_addr, cur->ai_addrlen) == 0)
        {
            break;
        }

        close(fd);
        fd = -1;
    }

    freeaddrinfo(addr_list);

    return fd;
}

// Read data from server
static int read_timeout(int sock, int epfd, unsigned char *buf, size_t len, uint32_t timeout)
{
    struct epoll_event ep_event;
    int ret = epoll_wait(epfd, &ep_event, 1, timeout);
    if (ret < 0)
    {
        fprintf(stderr, "Epoll_wait() returned %d, errno is %d\n", ret, errno);
    }
    else if (ret == 1)
    {
        ret = read(sock, buf, len);
        if (ret <= 0)
        {
            fprintf(stderr, "socket_read() returned %d, errno is %d\n", ret, errno);
        }
    }
    return ret;
}

// Open SoM device
static libusb_device_handle *open_device(uint16_t vid, uint16_t pid)
{
    libusb_device_handle *devh = NULL;
    int iofd = -1;
    int epfd = -1;

    // Open USB device
    devh = libusb_open_device_with_vid_pid(NULL, vid, pid);
    if (!devh)
    {
        fprintf(stderr, "Error opening SoM device\n");
        return NULL;
    }

    // Get the SoM device description
    libusb_device *dev = libusb_get_device(devh);
    struct libusb_device_descriptor desc = {0};
    libusb_get_device_descriptor(dev, &desc);

    // Check the SoM revision.
    // If the revision is 0x200, it is the Ear EVT device, and we need to switch to test authentication service
    const char *serviceUrl = (desc.bcdDevice == AzureEyeGen1Version) ? TestServiceUrl : PublicServiceUrl;
    iofd = new_net_connect(serviceUrl, ServicePort);
    if (iofd < 0)
    {
        fprintf(stderr, "Error connect to server\n");
        auth_pending = true;
        goto error;
    }

    // Epoll_wait for timeout
    epfd = epoll_create(1);

    struct epoll_event ep_event;
    ep_event.events = EPOLLIN;
    ep_event.data.fd = iofd;

    if (epoll_ctl(epfd, EPOLL_CTL_ADD, iofd, &ep_event) != 0)
    {
        fprintf(stderr, "Error epoll_ctl option\n");
        goto error;
    }

    som_info.devh = devh;
    som_info.iofd = iofd;
    som_info.epfd = epfd;
    memset(som_info.serial_data, 0, MAX_PACKAGE_SIZE);

    /* As we are dealing with a CDC-ACM device, it's highly probable that
        * Linux already attached the cdc-acm driver to this device.
        * We need to detach the drivers from all the USB interfaces. The CDC-ACM
        * Class defines two interfaces: the Control interface and the
        * Data interface.
        */
    for (int if_num = 0; if_num < 2; if_num++)
    {
        if (libusb_kernel_driver_active(som_info.devh, if_num))
        {
            libusb_detach_kernel_driver(som_info.devh, if_num);
        }
    }
    return devh;
error:
    if (epfd >= 0)
    {
        close(epfd);
    }
    if (iofd >= 0)
    {
        close(iofd);
    }
    if (devh)
    {
        libusb_close(devh);
    }
    return NULL;
}

// Close SoM device and release SOM_INFO resource
static void close_device()
{
    close(som_info.iofd);
    close(som_info.epfd);
    libusb_close(som_info.devh);
}

// Callback for USB bulk transfer
static void LIBUSB_CALL callbackUSBTransferComplete(struct libusb_transfer *xfr)
{
    struct libusb_transfer *xfr_next;
    int rc;
    int iofd = som_info.iofd;

    switch (xfr->status)
    {
    case LIBUSB_TRANSFER_COMPLETED:
        if (xfr->endpoint == EP_OUT_ADDRESS)
        {
#ifdef AUTH_LOG
            fprintf(stdout, "Sent TLS data to SoM:%d\n", xfr->actual_length);
#endif
            xfr_next = libusb_alloc_transfer(0);
            // Continue listening to the incoming data.
            // Before starting this step, SoM cannot send any data to validator through CDC.
            libusb_fill_bulk_transfer(xfr_next,
                                      xfr->dev_handle,
                                      EP_IN_ADDRESS,
                                      xfr->buffer,
                                      MAX_PACKAGE_SIZE,
                                      callbackUSBTransferComplete,
                                      NULL,
                                      0);
            rc = libusb_submit_transfer(xfr_next);
            if (rc < 0)
            {
                fprintf(stderr, "Error read connect response from SoM: %s\n", libusb_error_name(rc));
                libusb_free_transfer(xfr_next);
            }
        }
        // Receive data from SoM
        else if (xfr->endpoint == EP_IN_ADDRESS)
        {
            int len = xfr->actual_length;
#ifdef AUTH_LOG
            fprintf(stdout, "Received TLS data from SoM: %d\n", len);
#endif

            if (memcmp(xfr->buffer, PassAuthentication, 2) == 0)
            {
                fprintf(stdout, "Authentication Passed\n");
                som_close_required = true;
                auth_pass = true;
            }
            else if (memcmp(xfr->buffer, FailAuthentication, 2) == 0)
            {
                fprintf(stdout, "Authentication Failed\n");
                if(len > 2)
                {
                    fprintf(stdout, "%s\n", xfr->buffer + 2);
                }
                som_close_required = true;
                auth_pass = false;
            }
            // Transfer the data from SoM to service
            else if (memcmp(xfr->buffer, SendTLSDataToService, 2) == 0)
            {
                rc = write(iofd, xfr->buffer + 2, len - 2);
#ifdef AUTH_LOG
                fprintf(stdout, "Sent TLS data to service: %d\n", rc);
#endif

                // Continue listening to the incoming data.
                // Before starting this step, SoM cannot send any data to validator through CDC.
                xfr_next = libusb_alloc_transfer(0);
                libusb_fill_bulk_transfer(xfr_next,
                                          xfr->dev_handle,
                                          EP_IN_ADDRESS,
                                          xfr->buffer,
                                          MAX_PACKAGE_SIZE,
                                          callbackUSBTransferComplete,
                                          NULL,
                                          0);
                rc = libusb_submit_transfer(xfr_next);
                if (rc < 0)
                {
                    fprintf(stderr, "Error read connect response from SoM: %s\n", libusb_error_name(rc));
                    libusb_free_transfer(xfr_next);
                }
            }
            // Transfer the data from service to SoM
            else if (memcmp(xfr->buffer, ReceiveTLSDataFromService, 2) == 0)
            {
                unsigned int expectedLen;
                memcpy(&expectedLen, xfr->buffer + 2, 4);
#ifdef AUTH_LOG
                fprintf(stdout, "Tried to get TLS data from service: %d\n", expectedLen);
#endif
                int actualLen = 0;
                int epfd = som_info.epfd;

                if (expectedLen >= MAX_PACKAGE_SIZE)
                {
                    expectedLen = MAX_PACKAGE_SIZE;
                }

                actualLen = read_timeout(iofd, epfd, xfr->buffer, expectedLen, NETWORK_TIMEOUT);
                if (actualLen <= 0)
                {
                    fprintf(stderr, "Read_timeout error or timeout or eof\n");
                    auth_pending = true;
                    memset(xfr->buffer, 0, 10);
                    actualLen = 10;
                }
                else
                {
#ifdef AUTH_LOG
                    fprintf(stdout, "Get TLS data from Service: %d\n", actualLen);
#endif
                }

                // Send TLS data to SoM
                xfr_next = libusb_alloc_transfer(0);
                libusb_fill_bulk_transfer(xfr_next,
                                          xfr->dev_handle,
                                          EP_OUT_ADDRESS,
                                          xfr->buffer,
                                          actualLen,
                                          callbackUSBTransferComplete,
                                          NULL,
                                          0);
                rc = libusb_submit_transfer(xfr_next);
                if (rc < 0)
                {
                    fprintf(stderr, "Error sent TLS data to SoM: %s\n", libusb_error_name(rc));
                    libusb_free_transfer(xfr_next);
                }
            }
        }
        break;
    case LIBUSB_TRANSFER_CANCELLED:
    case LIBUSB_TRANSFER_NO_DEVICE:
    case LIBUSB_TRANSFER_TIMED_OUT:
    case LIBUSB_TRANSFER_ERROR:
    case LIBUSB_TRANSFER_STALL:
    case LIBUSB_TRANSFER_OVERFLOW:
        break;
    }
    libusb_free_transfer(xfr);
}

// Authenticate the SoM device
static void LIBUSB_CALL process_som_auth(libusb_device_handle *devh)
{
    // Send 0x7701 to SoM to keep SoM aware that Validator is ready for authentication process
    fprintf(stdout, "Start authentication\n");
    memcpy(som_info.serial_data, StartAuthentication, 2);
#ifdef AUTH_LOG
    fprintf(stdout, "Sent connect request to SoM\n");
#endif

    struct libusb_transfer *xfr;
    xfr = libusb_alloc_transfer(0);
    libusb_fill_bulk_transfer(xfr,
                              devh,
                              EP_OUT_ADDRESS,
                              som_info.serial_data,
                              strlen((char *)(som_info.serial_data)),
                              callbackUSBTransferComplete,
                              NULL,
                              0);
    int rc = libusb_submit_transfer(xfr);
    if (rc < 0)
    {
        fprintf(stderr, "Error sent service connect result to SoM: %s\n", libusb_error_name(rc));
        libusb_free_transfer(xfr);
    }
}

// Check whether a SoM is present
static bool check_device_existence(uint16_t vid, uint16_t pid)
{
    bool isExist = false;
    libusb_device_handle *devh = NULL;

    // check the attached SoM device
    devh = libusb_open_device_with_vid_pid(NULL, vid, pid);
    if (devh != NULL)
    {
        isExist = true;
        libusb_close(devh);
    }

    return isExist;
}

// Perform SoM authentication against the device
bool start_som_auth(uint16_t som_vid, uint16_t som_pid)
{
    som_close_required = false;
    auth_pass = false;
    int rc;
    rc = libusb_init(NULL);
    if (rc < 0)
    {
        fprintf(stderr, "Error initializing libusb: %s\n", libusb_error_name(rc));
        return false;
    }

    libusb_device_handle *devh = NULL;
    devh = open_device(som_vid, som_pid);

    if (devh == NULL)
        goto out;

    // Start SoM authentication process
    process_som_auth(devh);

    // Keep service running and monitor special usb device hotplug in or out.
    while (true)
    {
        rc = libusb_handle_events(NULL);
        if (rc < 0)
        {
            fprintf(stderr, "libusb_handle_events() failed: %s\n", libusb_error_name(rc));
        }

        if (!check_device_existence(som_vid, som_pid))
        {
            fprintf(stderr, "device has benn detched\n");
            goto close_device;
        }

        if (som_close_required)
        {
            goto close_device;
        }
    }

close_device:
    close_device();
out:
    libusb_exit(NULL);
    return auth_pass;
}

// Monitor the SoM device hotplug event, do authentication once SOM inserted
static void handle_hotplug_event(uint16_t som_vid, uint16_t som_pid)
{
    struct timeval time;
    struct udev *udev = udev_new();
    struct udev_monitor *monitor = udev_monitor_new_from_netlink(udev, "kernel");

    udev_monitor_filter_add_match_subsystem_devtype(monitor, "usb", NULL);
    udev_monitor_enable_receiving(monitor);

    int fd, ret;
    fd = udev_monitor_get_fd(monitor);
    fd_set readfds, tempfds;
    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);

    while (run)
    {
        if (auth_pending)
        {
            // Check whether the Internet is connected
            int iofd = new_net_connect(PublicServiceUrl, ServicePort);
            if (iofd >= 0)
            {
                close(iofd);
                auth_pending = false;
                start_som_auth(som_vid, som_pid);
            }
        }

        tempfds = readfds;
        time.tv_sec = 1;
        time.tv_usec = 0;

        ret = select(fd + 1, &tempfds, NULL, NULL, &time);

        if (ret <= 0)
        {
            continue;
        }
        if (FD_ISSET(fd, &tempfds))
        {
            struct udev_device *device = udev_monitor_receive_device(monitor);
            if (device == NULL)
                continue;

            const char *node = udev_device_get_devnode(device);
            const char *action = udev_device_get_action(device);

            if (node != NULL && action != NULL)
            {
                fprintf(stdout, "devnode: %s, action: %s\n", node, action);

                if (strcmp(action, "bind") == 0)
                {
                    const char *vid = udev_device_get_sysattr_value(device, "idVendor");
                    const char *pid = udev_device_get_sysattr_value(device, "idProduct");

                    if (vid != NULL && pid != NULL)
                    {
                        // Workaround for Eye SoM v2 authentication.
                        // If this is a Eye SoM v2 device, we set the authentication result to pass directly, and don't go through the authentication process
                        if (som_pid == AzureEyeGen1Pid && strtol(vid, NULL, 16) == Stm32CdcVid && strtol(pid, NULL, 16) == Stm32CdcPid)
                        {
                            const char *devclass = udev_device_get_sysattr_value(device, "bDeviceClass");
                            if (devclass != NULL && strtol(devclass, NULL, 16) == Stm32CdcClass)
                            {
                                fprintf(stdout, "Eye SoM v2 device is detected\n");
                                strncpy(som_node, node, sizeof(som_node) - 1);
                                auth_pass = true;
                            }
                        }
                        else if (strtol(vid, NULL, 16) == som_vid && strtol(pid, NULL, 16) == som_pid)
                        {
                            fprintf(stdout, "SoM device is bound, devnode is %s\n", node);
                            strncpy(som_node, node, sizeof(som_node) - 1);
                            start_som_auth(som_vid, som_pid);
                        }
                    }
                }
                else if (strcmp(action, "remove") == 0 && (strcmp(node, som_node) == 0))
                {
                    fprintf(stdout, "SoM device is removed, devnode is %s\n", node);
                    auth_pass = false;
                    auth_pending = false;
                }
            }

            udev_device_unref(device);
        }
    }
    udev_monitor_unref(monitor);
    udev_unref(udev);
}

// Check the existing SoM device and perform authenticaiton
static void handle_existing_device(uint16_t som_vid, uint16_t som_pid)
{
    struct udev *udev = udev_new();
    struct udev_enumerate *enumerator = udev_enumerate_new(udev);
    udev_enumerate_add_match_subsystem(enumerator, "usb");
    udev_enumerate_scan_devices(enumerator);

    struct udev_list_entry *entry = NULL;
    udev_list_entry_foreach(entry, udev_enumerate_get_list_entry(enumerator))
    {
        const char *const path = udev_list_entry_get_name(entry);
        struct udev_device *dev = udev_device_new_from_syspath(udev, path);
        if (dev == NULL)
            continue;

        const char *vid = udev_device_get_sysattr_value(dev, "idVendor");
        const char *pid = udev_device_get_sysattr_value(dev, "idProduct");
        const char *node = udev_device_get_devnode(dev);

        if (vid != NULL && pid != NULL && node != NULL)
        {
            // Workaround for Eye SoM v2 device authentication.
            // If this is a Eye SoM v2 device, we set the authentication result to pass directly, and don't go through the authentication process
            if (som_pid == AzureEyeGen1Pid && strtol(vid, NULL, 16) == Stm32CdcVid && strtol(pid, NULL, 16) == Stm32CdcPid)
            {
                const char *devclass = udev_device_get_sysattr_value(dev, "bDeviceClass");
                if (devclass != NULL && strtol(devclass, NULL, 16) == Stm32CdcClass)
                {
                    fprintf(stdout, "Eye SoM v2 device is detected\n");
                    strncpy(som_node, node, sizeof(som_node) - 1);
                    auth_pass = true;
                }
            }
            else if (strtol(vid, NULL, 16) == som_vid && strtol(pid, NULL, 16) == som_pid)
            {
                fprintf(stdout, "SoM device is bound, devnode is %s\n", node);
                strncpy(som_node, node, sizeof(som_node) - 1);
                start_som_auth(som_vid, som_pid);
            }
        }

        udev_device_unref(dev);
    }

    udev_enumerate_unref(enumerator);
    udev_unref(udev);
}

// Thread entry
static void *auth_entry(struct SOM_ID *som_id)
{
    handle_existing_device(som_id->vid, som_id->pid);

    handle_hotplug_event(som_id->vid, som_id->pid);

    return NULL;
}

// Start a new thread to perform SoM authentication
bool start_validator(uint16_t som_vid, uint16_t som_pid)
{
    som_id.vid = som_vid;
    som_id.pid = som_pid;
    run = true;

    int rc = pthread_create(&validator_thread_id, NULL, (void *(*)(void *))auth_entry, &som_id);
    if (rc != 0)
    {
        fprintf(stderr, "Failed to create validator thread: %s\n", strerror(rc));
        return false;
    }

    return true;
}

// Stop SoM authentication thread
int stop_validator()
{
    run = false;
    int rc = 0;
    rc = pthread_join(validator_thread_id, NULL);
    fprintf(stdout, "Validator is terminated\n");
    return rc;
}

// Get SoM authentication status
bool check_som_status()
{
    return auth_pass;
}
