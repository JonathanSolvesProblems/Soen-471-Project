###  article: boolean (nullable = true)
    - Keep: Yes 
    - Reason: Lets us know if an article was posted

###  body: string (nullable = true)
    - Keep: Yes
    - Reason: Gives us the text content of the post

### bodywithurls: string (nullable = true)
    - Keep: No
    - Reason: We will utilize url_domains to get the URLs if something is linked

### color: string (nullable = true)
    - Keep: No
    - Reason: Not useful to this dataset

### commentDepth: long (nullable = true)
    - Keep: No
    - Reason: For datatype: comment
    
### comments: long (nullable = true)
    - Keep: Yes
    - Reason: Promising feature

### controversy: double (nullable = true)
    - Keep: ?
    - Reason: Discuss / Can be useful

### conversation: string (nullable = true)
    - Keep: ?
    - Reason: ?

### createdAt: string (nullable = true)
    - Keep: Depends
    - Reason: Do we want to keep this or formatted version?

### createdAtformatted: string (nullable = true)
    - Keep: Depends
    - Reason: Do we want to keep this or EpochTimestamp version?

### creator: string (nullable = true)
    - Keep: Depends
    - Do we want to use the User's real handle or their randomly generated Id String

### datatype: string (nullable = true)
    - Keep: Yes
    - Reason: Need to filter by Posts

### depth: long (nullable = true)
    - Keep: No
    - Reason: For comments

### depthRaw: long (nullable = true)
    - Keep: No
    - Reason: For comments

### downvotes: string (nullable = true)
    - Keep: No
    - Reason: Posts do not have downvotes. Similar to 'likes' on Facebook

### followers: long (nullable = true)
    - Keep: Yes
    - Reason: Good indicator of upvotes

### following: long (nullable = true)
    - Keep: Yes
    - Reason: Good indicator of upvotes

### hashtags: string (nullable = true)
    - Keep: Yes
    - Reason: Somewhat of a good indicator of upvotes

### id: string (nullable = true)
    - Keep: Yes
    - Reason: Need unique identifier for posts

### impressions: long (nullable = true)
    - Keep: Yes
    - Reason: Good indicator of upvotes

### isPrimary: boolean (nullable = true)
    - Keep: Maybe? 
    - Reason: I believe it shows if its a primary account? Not too sure what it means

### lastseents: string (nullable = true)
    - Keep: No
    - Reason: Not needed

### links: string (nullable = true)
    - Keep: No
    - Reason: Gives an array of random letters 
    (Im assuming its somehow leads to a URL link somehow)

### media: long (nullable = true)
    - Keep: No
    - Reason: Random numbers (I dont know what it means) Could 
    be the amount of times its shared on other places of media.

### parent: string (nullable = true)
    - Keep: No
    - Reason: Shows the parent id of comment/post (More useful for comments, not for our use case)

### post: string (nullable = true)
    - Keep: No
    - Reason: For comment datatypes, Im assuming its to link to the post from a comment

### posts: long (nullable = true)
    - Keep: ?
    - Reason: The amount of posts that the user has made at the time of being scrapped. Doesnt show
    how many posts the user has made at the time post was created. Only when it was scrapped. 

### preview: string (nullable = true)
    - Keep: No
    - Reason: Redundant data showing preview of post

### replyingTo: string (nullable = true)
    - Keep: No
    - Reason: For comments

### reposts: long (nullable = true)
    - Keep: Yes
    - Reason: The amount of time something a post was reposted

### score: long (nullable = true)
    - Keep: No
    - Reason: Score is only used for comments

### sensitive: boolean (nullable = true)
    - Keep: ?
    - Reason: Could be a factor, if a post is marked as sensitive or not. 
    (I think its similar to facebook showing a warning screen saying 'this post may be sensitive to some users')

### state: long (nullable = true)
    - Keep: No
    - Reason: Doesnt give us much information. Only gives us a number

### upvotes: long (nullable = true)
    - Keep: Yes
    - Reason: This is our prediction label

### username: string (nullable = true)
    - Keep: Yes
    - Reason: Author of post is good indicator

### verified: boolean (nullable = true)
    - Keep: Yes
    - Reason: Lets us know if an account is verified

### urls_createdAt: string (nullable = true)
    - Keep: No
    - Reason: Not useful/duplicate of createdAt timestamp

### urls_domain: string (nullable = true)
    - Keep: Yes
    - Reason: Lets us know what domains are being linked

### urls_id: string (nullable = true)
    - Keep: No
    - Reason: Doesn't give us much information. Why does URLs contain an ID? 

### urls_long: string (nullable = true)
    - Keep: ?
    - Reason: Gives us the full URL of what was linked

### urls_modified: string (nullable = true)
    - Keep: Not needed
    - Reason: Modifies URLs_long when posting the link, removes 'https://' for example

### urls_short: string (nullable = true)
    - Keep: Not needed
    - Reason: Gives us shortened version of URL that was linked

### urls_state: string (nullable = true)
    - Keep: No
    - Reason: Not really relevant to what we need. Assuming its talking about the current state of the URL linked

### urls_metadata_length: long (nullable = true)
    - Keep: ?
    - Reason: Maybe, can be a contributor 

### urls_metadata_mimeType: string (nullable = true)
    - Keep: ?
    - Reason: Lets us know if its an text/html or a video

### urls_metadata_site: string (nullable = true)
    - Keep: Yes
    - Reason: Can tell us what domain its from, however there are some issues as some self-posted 
    posts such as Images might not include this attribute. So we might need to combine with other attribute to get