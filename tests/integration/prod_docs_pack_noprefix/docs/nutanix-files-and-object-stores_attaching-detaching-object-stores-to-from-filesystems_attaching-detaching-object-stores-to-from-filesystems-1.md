---
description:
---

# Attach or detach object store buckets using the CLI

Using the CLI, you can:

* Attach an object store bucket to a filesystem
* Detach an object store bucket from a filesystem

## **Attach an object store bucket** to a filesystem

**Command:** `ncli files object-store attach`

To attach an object store to a filesystem, use the following command:

`ncli files object-store attach share=<share-name> object-store=<object-store-name> mode=<mode>`

**Parameters**

 | Name | Value | Default |
 | --- | --- | --- |
 | fs-name* | Name of the filesystem to attach with the object store. | ​ |
 | obs-name* | Name of the object store to attach. |  |
 | mode | The operational mode for the object store bucket.The possible values are:writable: Local access for read/write operations.remote: Read-only access for remote object stores. | writable |

## **Detach an object store bucket** from a filesystem

**Command:** `ncli files object-store detach`

To detach an object store from a filesystem, use the following command:

`ncli files object-store detach share=<share-name> object-store=<object-store-name>`

**Parameters**

 | Name | Value |
 | --- | --- |
 | fs-name* | Name of the filesystem to be detached from the object store |
 | obs-name* | Name of the object store to be detached |

Note: To [recover from a snapshot](../../snap-to-obj#creating-a-filesystem-from-a-snapshot-using-the-cli) uploaded when two `local` object stores have been attached, use the `additional-object-store` parameter in the `ncli files snapshot restore` command. The primary object store should be the one where the locator has been uploaded to
